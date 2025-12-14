import { useState } from 'react'
import { useAuth } from '../context/AuthContext'
import { Save, User, Key, Shield, Bell } from 'lucide-react'
import clsx from 'clsx'

type TabId = 'profile' | 'security' | 'notifications'

interface Tab {
  id: TabId
  name: string
  icon: typeof User
}

const tabs: Tab[] = [
  { id: 'profile', name: 'Profile', icon: User },
  { id: 'security', name: 'Security', icon: Shield },
  { id: 'notifications', name: 'Notifications', icon: Bell },
]

function ProfileTab() {
  const { user } = useAuth()
  const [email, setEmail] = useState(user?.email || '')
  const [isSaving, setIsSaving] = useState(false)

  const handleSave = async () => {
    setIsSaving(true)
    // TODO: Implement profile update
    await new Promise((resolve) => setTimeout(resolve, 1000))
    setIsSaving(false)
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900">Profile Information</h3>
        <p className="mt-1 text-sm text-gray-500">
          Update your account information and email address.
        </p>
      </div>

      <div className="space-y-4">
        <div>
          <label className="label">Username</label>
          <input
            type="text"
            value={user?.username || ''}
            disabled
            className="input bg-gray-50"
          />
          <p className="mt-1 text-xs text-gray-500">Username cannot be changed</p>
        </div>

        <div>
          <label className="label">Email Address</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="input"
          />
        </div>

        <div>
          <label className="label">Account Type</label>
          <div className="flex items-center space-x-2">
            <span className={clsx(
              'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
              user?.is_admin ? 'bg-purple-100 text-purple-800' : 'bg-gray-100 text-gray-800'
            )}>
              {user?.is_admin ? 'Administrator' : 'User'}
            </span>
          </div>
        </div>

        <div>
          <label className="label">Member Since</label>
          <p className="text-sm text-gray-900">
            {user?.created_at
              ? new Date(user.created_at).toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric',
                })
              : 'Unknown'}
          </p>
        </div>
      </div>

      <div className="pt-4 border-t border-gray-200">
        <button
          onClick={handleSave}
          disabled={isSaving}
          className="btn btn-primary"
        >
          {isSaving ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
              Saving...
            </>
          ) : (
            <>
              <Save className="h-4 w-4 mr-2" />
              Save Changes
            </>
          )}
        </button>
      </div>
    </div>
  )
}

function SecurityTab() {
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState('')

  const handleChangePassword = async () => {
    setError('')

    if (newPassword !== confirmPassword) {
      setError('New passwords do not match')
      return
    }

    if (newPassword.length < 8) {
      setError('Password must be at least 8 characters')
      return
    }

    setIsSaving(true)
    // TODO: Implement password change
    await new Promise((resolve) => setTimeout(resolve, 1000))
    setIsSaving(false)
    setCurrentPassword('')
    setNewPassword('')
    setConfirmPassword('')
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900">Change Password</h3>
        <p className="mt-1 text-sm text-gray-500">
          Update your password to keep your account secure.
        </p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-700">
          {error}
        </div>
      )}

      <div className="space-y-4">
        <div>
          <label className="label">Current Password</label>
          <input
            type="password"
            value={currentPassword}
            onChange={(e) => setCurrentPassword(e.target.value)}
            className="input"
          />
        </div>

        <div>
          <label className="label">New Password</label>
          <input
            type="password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            className="input"
          />
        </div>

        <div>
          <label className="label">Confirm New Password</label>
          <input
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            className="input"
          />
        </div>
      </div>

      <div className="pt-4 border-t border-gray-200">
        <button
          onClick={handleChangePassword}
          disabled={isSaving || !currentPassword || !newPassword || !confirmPassword}
          className="btn btn-primary"
        >
          {isSaving ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
              Updating...
            </>
          ) : (
            <>
              <Key className="h-4 w-4 mr-2" />
              Update Password
            </>
          )}
        </button>
      </div>
    </div>
  )
}

function NotificationsTab() {
  const [emailNotifications, setEmailNotifications] = useState(true)
  const [jobComplete, setJobComplete] = useState(true)
  const [jobFailed, setJobFailed] = useState(true)

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900">Notification Preferences</h3>
        <p className="mt-1 text-sm text-gray-500">
          Choose how you want to be notified about training jobs.
        </p>
      </div>

      <div className="space-y-4">
        <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
          <div>
            <span className="text-sm font-medium text-gray-900">Email Notifications</span>
            <p className="text-xs text-gray-500">
              Receive email updates about your training jobs
            </p>
          </div>
          <button
            onClick={() => setEmailNotifications(!emailNotifications)}
            className={clsx(
              'relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out',
              emailNotifications ? 'bg-primary-600' : 'bg-gray-200'
            )}
          >
            <span
              className={clsx(
                'pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out',
                emailNotifications ? 'translate-x-5' : 'translate-x-0'
              )}
            />
          </button>
        </div>

        <div className="pl-4 space-y-3">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={jobComplete}
              onChange={(e) => setJobComplete(e.target.checked)}
              disabled={!emailNotifications}
              className="rounded border-gray-300 text-primary-600 focus:ring-primary-500 disabled:opacity-50"
            />
            <span className={clsx(
              'ml-2 text-sm',
              emailNotifications ? 'text-gray-900' : 'text-gray-400'
            )}>
              Job completed successfully
            </span>
          </label>

          <label className="flex items-center">
            <input
              type="checkbox"
              checked={jobFailed}
              onChange={(e) => setJobFailed(e.target.checked)}
              disabled={!emailNotifications}
              className="rounded border-gray-300 text-primary-600 focus:ring-primary-500 disabled:opacity-50"
            />
            <span className={clsx(
              'ml-2 text-sm',
              emailNotifications ? 'text-gray-900' : 'text-gray-400'
            )}>
              Job failed or cancelled
            </span>
          </label>
        </div>
      </div>

      <div className="pt-4 border-t border-gray-200">
        <button className="btn btn-primary">
          <Save className="h-4 w-4 mr-2" />
          Save Preferences
        </button>
      </div>
    </div>
  )
}

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<TabId>('profile')

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-900 mb-6">Settings</h1>

      <div className="card">
        <div className="border-b border-gray-200">
          <nav className="flex -mb-px">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={clsx(
                  'flex items-center px-6 py-4 text-sm font-medium border-b-2 transition-colors',
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                )}
              >
                <tab.icon className="h-5 w-5 mr-2" />
                {tab.name}
              </button>
            ))}
          </nav>
        </div>

        <div className="p-6">
          {activeTab === 'profile' && <ProfileTab />}
          {activeTab === 'security' && <SecurityTab />}
          {activeTab === 'notifications' && <NotificationsTab />}
        </div>
      </div>
    </div>
  )
}
