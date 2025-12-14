"""Training job orchestrator for managing training execution."""

import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from largeforge.config import TrainingConfig
from largeforge.training.base import BaseTrainer, TrainingCallback, TrainingState
from largeforge.training.events import EventEmitter, EventType, TrainingEvent, emit_event
from largeforge.training.progress import ProgressCallback, ProgressTracker
from largeforge.utils import get_logger, ensure_dir
from largeforge.web.schemas import TrainingProgress
from largeforge.web.state import JobStateManager, JobStatus, TrainingJob

logger = get_logger(__name__)


class TrainingOrchestrator:
    """
    Orchestrates training job execution.

    Bridges the web layer with training modules, handling job lifecycle,
    callbacks, and real-time progress updates.
    """

    def __init__(
        self,
        state_manager: JobStateManager,
        connection_manager: Optional[Any] = None,
        max_concurrent: int = 1,
    ):
        """
        Initialize training orchestrator.

        Args:
            state_manager: Job state manager
            connection_manager: WebSocket connection manager for real-time updates
            max_concurrent: Maximum concurrent training jobs
        """
        self.state_manager = state_manager
        self.connection_manager = connection_manager
        self.max_concurrent = max_concurrent

        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix="training-"
        )
        self._running_jobs: Dict[str, Future] = {}
        self._stop_flags: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

        logger.info(f"TrainingOrchestrator initialized (max_concurrent={max_concurrent})")

    def start_job(self, job: TrainingJob) -> bool:
        """
        Start a training job.

        Args:
            job: TrainingJob to start

        Returns:
            True if job was started, False otherwise
        """
        with self._lock:
            if job.job_id in self._running_jobs:
                logger.warning(f"Job {job.job_id} is already running")
                return False

            if len(self._running_jobs) >= self.max_concurrent:
                logger.warning(
                    f"Cannot start job {job.job_id}: max concurrent jobs reached"
                )
                return False

            # Create stop flag
            stop_flag = threading.Event()
            self._stop_flags[job.job_id] = stop_flag

            # Submit to executor
            future = self._executor.submit(self._run_training, job, stop_flag)
            self._running_jobs[job.job_id] = future

            logger.info(f"Started job {job.job_id}")
            return True

    def stop_job(self, job_id: str) -> bool:
        """
        Stop a running training job.

        Args:
            job_id: Job ID to stop

        Returns:
            True if job was stopped, False if not running
        """
        with self._lock:
            if job_id not in self._running_jobs:
                return False

            # Set stop flag
            if job_id in self._stop_flags:
                self._stop_flags[job_id].set()

            # Update state
            self.state_manager.cancel_job(job_id)

            # Emit event
            emit_event(EventType.JOB_CANCELLED, job_id)

            logger.info(f"Stopped job {job_id}")
            return True

    def get_running_jobs(self) -> List[str]:
        """Get list of running job IDs."""
        with self._lock:
            return list(self._running_jobs.keys())

    def get_running_count(self) -> int:
        """Get number of running jobs."""
        with self._lock:
            return len(self._running_jobs)

    def can_start_job(self) -> bool:
        """Check if a new job can be started."""
        with self._lock:
            return len(self._running_jobs) < self.max_concurrent

    def _run_training(self, job: TrainingJob, stop_flag: threading.Event) -> None:
        """
        Execute training job in background thread.

        Args:
            job: TrainingJob to execute
            stop_flag: Event to signal job cancellation
        """
        job_id = job.job_id

        try:
            logger.info(f"Starting training for job {job_id}")

            # Update state to running
            self.state_manager.set_running(job_id)
            emit_event(EventType.JOB_STARTED, job_id)

            # Setup output directory
            output_dir = job.config.get("output_dir")
            if not output_dir:
                output_dir = str(Path(".largeforge/outputs") / job_id)
            ensure_dir(output_dir)

            # Create trainer and run
            trainer = self._create_trainer(job, output_dir, stop_flag)
            callbacks = self._create_callbacks(job, stop_flag)

            for callback in callbacks:
                trainer.add_callback(callback)

            # Run training
            state = trainer.train()

            # Check if stopped
            if stop_flag.is_set():
                logger.info(f"Job {job_id} was cancelled")
                return

            # Mark as completed
            self.state_manager.complete_job(job_id, output_dir)
            emit_event(
                EventType.JOB_COMPLETED,
                job_id,
                {"output_path": output_dir}
            )

            logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Job {job_id} failed: {error_msg}")
            logger.debug(traceback.format_exc())

            self.state_manager.fail_job(job_id, error_msg)
            emit_event(
                EventType.JOB_FAILED,
                job_id,
                {"error": error_msg}
            )

        finally:
            # Cleanup
            with self._lock:
                self._running_jobs.pop(job_id, None)
                self._stop_flags.pop(job_id, None)

    def _create_trainer(
        self,
        job: TrainingJob,
        output_dir: str,
        stop_flag: threading.Event,
    ) -> BaseTrainer:
        """
        Create appropriate trainer for job.

        Args:
            job: TrainingJob
            output_dir: Output directory
            stop_flag: Cancellation flag

        Returns:
            Configured trainer instance
        """
        config = job.config
        training_type = job.training_type

        # Build training config
        training_config = TrainingConfig(
            output_dir=output_dir,
            num_train_epochs=config.get("epochs", 3),
            per_device_train_batch_size=config.get("batch_size", 4),
            learning_rate=config.get("learning_rate", 2e-5),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            warmup_ratio=config.get("warmup_ratio", 0.1),
            weight_decay=config.get("weight_decay", 0.01),
            max_grad_norm=config.get("max_grad_norm", 1.0),
            fp16=config.get("fp16", False),
            bf16=config.get("bf16", True),
            logging_steps=config.get("logging_steps", 10),
            save_steps=config.get("save_steps", 500),
            eval_steps=config.get("eval_steps", 500),
        )

        # Load model and tokenizer
        model, tokenizer = self._load_model_and_tokenizer(
            job.model_path,
            config.get("lora_enabled", True),
            config.get("lora_r", 16),
            config.get("lora_alpha", 32),
            config.get("lora_dropout", 0.05),
        )

        # Load dataset
        train_dataset, eval_dataset = self._load_datasets(
            job.dataset_path,
            training_type,
            tokenizer,
            config.get("max_seq_length", 2048),
        )

        # Create trainer based on type
        if training_type == "sft":
            from largeforge.training.sft import SFTTrainer
            trainer = SFTTrainer(
                model=model,
                config=training_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
            )
        elif training_type == "dpo":
            from largeforge.training.dpo import DPOTrainer
            trainer = DPOTrainer(
                model=model,
                config=training_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                beta=config.get("dpo_beta", 0.1),
            )
        else:
            raise ValueError(f"Unknown training type: {training_type}")

        return trainer

    def _load_model_and_tokenizer(
        self,
        model_path: str,
        lora_enabled: bool,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        """Load model and tokenizer with optional LoRA."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info(f"Loading model from {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Determine device and dtype
        device_map = "auto"
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        # Apply LoRA if enabled
        if lora_enabled:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        return model, tokenizer

    def _load_datasets(
        self,
        dataset_path: str,
        training_type: str,
        tokenizer,
        max_seq_length: int,
    ):
        """Load and prepare datasets."""
        from largeforge.data import load_sft_dataset, load_dpo_dataset

        logger.info(f"Loading dataset from {dataset_path}")

        if training_type == "sft":
            train_dataset = load_sft_dataset(
                dataset_path,
                tokenizer=tokenizer,
                max_length=max_seq_length,
            )
        elif training_type == "dpo":
            train_dataset = load_dpo_dataset(
                dataset_path,
                tokenizer=tokenizer,
                max_length=max_seq_length,
            )
        else:
            raise ValueError(f"Unknown training type: {training_type}")

        # Use 10% for evaluation if no separate eval set
        eval_dataset = None

        return train_dataset, eval_dataset

    def _create_callbacks(
        self,
        job: TrainingJob,
        stop_flag: threading.Event,
    ) -> List[TrainingCallback]:
        """
        Create callbacks for training job.

        Args:
            job: TrainingJob
            stop_flag: Cancellation flag

        Returns:
            List of training callbacks
        """
        callbacks = []
        job_id = job.job_id

        # Progress callback with WebSocket broadcast
        def on_progress(tracker: ProgressTracker):
            # Update state manager
            progress = TrainingProgress(
                epoch=tracker.epoch,
                total_epochs=tracker.total_epochs,
                step=tracker.step,
                total_steps=tracker.total_steps,
                loss=tracker.loss,
                learning_rate=tracker.learning_rate,
                elapsed_seconds=tracker.elapsed_seconds,
                eta_seconds=tracker.eta_seconds,
                metrics=tracker.metrics,
            )
            self.state_manager.update_progress(job_id, progress)

            # Broadcast via WebSocket if available
            if self.connection_manager:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(
                            self.connection_manager.broadcast_progress(job_id, progress)
                        )
                except RuntimeError:
                    pass

        progress_callback = ProgressCallback(
            callback_fn=on_progress,
            update_interval=job.config.get("logging_steps", 10),
        )
        callbacks.append(progress_callback)

        # Stop check callback
        callbacks.append(StopCheckCallback(stop_flag))

        # Event emitter callback
        callbacks.append(EventEmitterCallback(job_id))

        return callbacks

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the orchestrator.

        Args:
            wait: Wait for running jobs to complete
        """
        logger.info("Shutting down TrainingOrchestrator")

        # Signal all jobs to stop
        with self._lock:
            for stop_flag in self._stop_flags.values():
                stop_flag.set()

        # Shutdown executor
        self._executor.shutdown(wait=wait)


class StopCheckCallback(TrainingCallback):
    """Callback that checks for job cancellation."""

    def __init__(self, stop_flag: threading.Event):
        self.stop_flag = stop_flag

    def on_step_end(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        """Check stop flag after each step."""
        if self.stop_flag.is_set():
            state.should_stop = True


class EventEmitterCallback(TrainingCallback):
    """Callback that emits training events."""

    def __init__(self, job_id: str):
        self.job_id = job_id

    def on_epoch_begin(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        emit_event(
            EventType.EPOCH_START,
            self.job_id,
            {"epoch": state.epoch}
        )

    def on_epoch_end(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        emit_event(
            EventType.EPOCH_END,
            self.job_id,
            {"epoch": state.epoch}
        )

    def on_step_end(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        if state.global_step % 100 == 0:  # Don't emit for every step
            emit_event(
                EventType.STEP_COMPLETE,
                self.job_id,
                {"step": state.global_step}
            )

    def on_save(
        self, trainer: BaseTrainer, state: TrainingState, **kwargs
    ) -> None:
        emit_event(
            EventType.CHECKPOINT_SAVED,
            self.job_id,
            {"step": state.global_step}
        )

    def on_evaluate(
        self, trainer: BaseTrainer, state: TrainingState, metrics: Dict[str, float], **kwargs
    ) -> None:
        emit_event(
            EventType.EVALUATION_COMPLETE,
            self.job_id,
            {"metrics": metrics}
        )
