import { useState, useEffect, useCallback } from 'react';
import { useAuthStore } from '../store/authStore';
import { getAlerts, updateAlert } from '../api/alertsApi';

// ── Helpers ───────────────────────────────────────────────────
function getRiskColor(score) {
  if (score >= 0.85) return { text: 'text-red-400', bg: 'bg-red-500/10 border-red-500/20' };
  if (score >= 0.75)
    return { text: 'text-orange-400', bg: 'bg-orange-500/10 border-orange-500/20' };
  return { text: 'text-yellow-400', bg: 'bg-yellow-500/10 border-yellow-500/20' };
}

function getStatusStyle(status) {
  switch (status) {
    case 'OPEN':
      return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20';
    case 'UNDER_REVIEW':
      return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
    case 'CONFIRMED':
      return 'bg-red-500/10 text-red-400 border-red-500/20';
    case 'DISMISSED':
      return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
    default:
      return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
  }
}

function formatDate(iso) {
  return new Date(iso).toLocaleDateString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

// ── Alert Row (expandable) ────────────────────────────────────
function AlertRow({ alert, canEdit, onUpdate }) {
  const [expanded, setExpanded] = useState(false);
  const [editStatus, setEditStatus] = useState(alert.status);
  const [editNotes, setEditNotes] = useState(alert.notes || '');
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState('');
  const [saved, setSaved] = useState(false);

  const risk = getRiskColor(alert.risk_score);
  const isFinal = alert.status === 'CONFIRMED' || alert.status === 'DISMISSED';

  const handleSave = async () => {
    setSaving(true);
    setSaveError('');
    try {
      const updated = await updateAlert(alert.id, {
        status: editStatus,
        notes: editNotes || null,
      });
      onUpdate(updated);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (err) {
      setSaveError(err.response?.data?.detail || 'Failed to save changes.');
    } finally {
      setSaving(false);
    }
  };

  const hasChanges = editStatus !== alert.status || editNotes !== (alert.notes || '');

  return (
    <div className="border border-gray-800 rounded-xl overflow-hidden mb-3 transition-all duration-200">
      {/* Row Header — always visible */}
      <div
        className="flex items-center gap-4 px-5 py-4 cursor-pointer hover:bg-gray-800/40 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        {/* Risk Score */}
        <div className={`border rounded-lg px-3 py-1.5 text-center min-w-[64px] ${risk.bg}`}>
          <p className={`text-sm font-bold ${risk.text}`}>{(alert.risk_score * 100).toFixed(0)}%</p>
          <p className="text-gray-600 text-xs">risk</p>
        </div>

        {/* Main Info */}
        <div className="flex-1 min-w-0">
          <p className="text-white text-sm font-medium truncate">{alert.reason}</p>
          <p className="text-gray-600 text-xs mt-0.5">Created {formatDate(alert.created_at)}</p>
        </div>

        {/* Status Badge */}
        <span
          className={`border rounded-full px-3 py-1 text-xs font-medium flex-shrink-0 ${getStatusStyle(alert.status)}`}
        >
          {alert.status.replace('_', ' ')}
        </span>

        {/* Expand Arrow */}
        <span
          className={`text-gray-600 text-xs transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
        >
          ▼
        </span>
      </div>

      {/* Expanded Detail Panel */}
      {expanded && (
        <div className="border-t border-gray-800 bg-gray-900/50 px-5 py-5">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left — Alert Details */}
            <div>
              <h4 className="text-xs font-medium text-gray-500 uppercase tracking-widest mb-4">
                Alert Details
              </h4>
              <div className="space-y-3">
                <div className="flex justify-between items-start">
                  <span className="text-gray-500 text-xs">Alert ID</span>
                  <span className="text-gray-400 text-xs font-mono">{alert.id.slice(0, 8)}...</span>
                </div>
                <div className="flex justify-between items-start">
                  <span className="text-gray-500 text-xs">Transaction ID</span>
                  <span className="text-gray-400 text-xs font-mono">
                    {alert.transaction_id.slice(0, 8)}...
                  </span>
                </div>
                <div className="flex justify-between items-start">
                  <span className="text-gray-500 text-xs">Risk Score</span>
                  <span className={`text-xs font-semibold ${risk.text}`}>{alert.risk_score}</span>
                </div>
                <div className="flex justify-between items-start">
                  <span className="text-gray-500 text-xs">Last Updated</span>
                  <span className="text-gray-400 text-xs">{formatDate(alert.updated_at)}</span>
                </div>
                {alert.notes && (
                  <div className="pt-2 border-t border-gray-800">
                    <span className="text-gray-500 text-xs block mb-1">Notes</span>
                    <p className="text-gray-300 text-xs leading-relaxed">{alert.notes}</p>
                  </div>
                )}
              </div>
            </div>

            {/* Right — Edit Panel */}
            <div>
              <h4 className="text-xs font-medium text-gray-500 uppercase tracking-widest mb-4">
                {canEdit && !isFinal ? 'Update Alert' : 'Status'}
              </h4>

              {/* Can edit and not final */}
              {canEdit && !isFinal ? (
                <div className="space-y-4">
                  <div>
                    <label className="block text-xs text-gray-500 mb-2">Status</label>
                    <select
                      value={editStatus}
                      onChange={(e) => setEditStatus(e.target.value)}
                      className="w-full bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-3 py-2.5 focus:outline-none focus:border-green-500"
                    >
                      <option value="OPEN">Open</option>
                      <option value="UNDER_REVIEW">Under Review</option>
                      <option value="CONFIRMED">Confirmed</option>
                      <option value="DISMISSED">Dismissed</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 mb-2">Notes</label>
                    <textarea
                      value={editNotes}
                      onChange={(e) => setEditNotes(e.target.value)}
                      rows={3}
                      placeholder="Add investigation notes..."
                      className="w-full bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-3 py-2.5 focus:outline-none focus:border-green-500 resize-none placeholder-gray-600"
                    />
                  </div>
                  {saveError && <p className="text-red-400 text-xs">{saveError}</p>}
                  <button
                    onClick={handleSave}
                    disabled={saving || !hasChanges}
                    className="w-full bg-green-500 hover:bg-green-400 disabled:opacity-40 disabled:cursor-not-allowed text-black font-semibold text-sm rounded-lg py-2.5 transition-colors"
                  >
                    {saving ? 'Saving...' : saved ? '✓ Saved' : 'Save Changes'}
                  </button>
                </div>
              ) : (
                // Read-only view
                <div className="space-y-3">
                  <div
                    className={`border rounded-lg px-4 py-3 inline-block ${getStatusStyle(alert.status)}`}
                  >
                    <p className="text-sm font-medium">{alert.status.replace('_', ' ')}</p>
                  </div>
                  {isFinal && canEdit && (
                    <p className="text-gray-600 text-xs">
                      This alert has been resolved and cannot be edited.
                    </p>
                  )}
                  {!canEdit && (
                    <p className="text-gray-600 text-xs">You have read-only access to alerts.</p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main Alerts Page ──────────────────────────────────────────
function AlertsPage() {
  const { user } = useAuthStore();
  const canEdit = user?.role === 'ADMIN' || user?.role === 'ANALYST';

  const [alerts, setAlerts] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [page, setPage] = useState(0);
  const pageSize = 10;

  const fetchAlerts = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = {
        skip: page * pageSize,
        limit: pageSize,
      };
      // Server-side filter: ask the backend for just the requested tab. This
      // makes pagination accurate (otherwise a page of 10 OPEN alerts can
      // hide more behind DISMISSED rows that filled the same page).
      if (statusFilter !== 'ALL') {
        params.state = statusFilter;
      }
      const data = await getAlerts(params);
      setAlerts(data.items);
      setTotal(data.total);
    } catch (err) {
      setError('Failed to load alerts. Please try again.');
    } finally {
      setLoading(false);
    }
  }, [page, statusFilter]);

  useEffect(() => {
    fetchAlerts();
  }, [fetchAlerts]);

  useEffect(() => {
    // Reset to page 1 when the user switches tabs so the previous page index
    // doesn't survive a tab change with fewer rows.
    setPage(0);
  }, [statusFilter]);

  // Called when an alert is updated — replaces the old version in state
  const handleUpdate = (updatedAlert) => {
    setAlerts((prev) => prev.map((a) => (a.id === updatedAlert.id ? updatedAlert : a)));
  };

  // The backend already filters by tab. Counts in tabs reflect the count of
  // the currently fetched tab (and total for ALL) so users still see a number
  // next to the active tab without an extra round-trip per tab.
  const filteredAlerts = alerts;
  const tabs = [
    { key: 'ALL', label: 'All', count: statusFilter === 'ALL' ? total : '' },
    { key: 'OPEN', label: 'Open', count: statusFilter === 'OPEN' ? total : '' },
    {
      key: 'UNDER_REVIEW',
      label: 'Under Review',
      count: statusFilter === 'UNDER_REVIEW' ? total : '',
    },
    {
      key: 'CONFIRMED',
      label: 'Confirmed',
      count: statusFilter === 'CONFIRMED' ? total : '',
    },
    {
      key: 'DISMISSED',
      label: 'Dismissed',
      count: statusFilter === 'DISMISSED' ? total : '',
    },
  ];

  const totalPages = Math.ceil(total / pageSize);

  // ── Loading ─────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-green-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-500 text-sm">Loading alerts...</p>
        </div>
      </div>
    );
  }

  // ── Error ───────────────────────────────────────────────────
  if (error) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-8 text-center max-w-md">
          <p className="text-red-400 text-sm">{error}</p>
          <button
            onClick={fetchAlerts}
            className="mt-4 text-xs text-gray-400 hover:text-white underline"
          >
            Try again
          </button>
        </div>
      </div>
    );
  }

  // ── Page ────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <div className="max-w-6xl mx-auto px-6 py-10">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-white mb-1">Alerts</h1>
          <p className="text-gray-500 text-sm">
            {total} total alert{total !== 1 ? 's' : ''} — sorted by risk score
            {canEdit ? ' · Click any alert to review and update its status' : ' · Read-only view'}
          </p>
        </div>

        {/* Status Filter Tabs */}
        <div className="flex gap-2 mb-6 flex-wrap">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setStatusFilter(tab.key)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium border transition-colors ${
                statusFilter === tab.key
                  ? 'bg-green-500/10 border-green-500/30 text-green-400'
                  : 'bg-gray-900 border-gray-800 text-gray-400 hover:border-gray-700 hover:text-gray-300'
              }`}
            >
              {tab.label}
              <span
                className={`rounded-full px-2 py-0.5 text-xs ${
                  statusFilter === tab.key ? 'bg-green-500/20' : 'bg-gray-800'
                }`}
              >
                {tab.count}
              </span>
            </button>
          ))}
        </div>

        {/* Alert List */}
        {filteredAlerts.length === 0 ? (
          <div className="text-center py-20 border border-gray-800 rounded-xl">
            <p className="text-gray-500 text-sm">No alerts found for this filter.</p>
          </div>
        ) : (
          filteredAlerts.map((alert) => (
            <AlertRow key={alert.id} alert={alert} canEdit={canEdit} onUpdate={handleUpdate} />
          ))
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between mt-8">
            <p className="text-gray-600 text-xs">
              Page {page + 1} of {totalPages} · {total} total alerts
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="px-4 py-2 bg-gray-900 border border-gray-800 text-gray-400 text-xs rounded-lg disabled:opacity-40 hover:border-gray-700 transition-colors"
              >
                Previous
              </button>
              <button
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="px-4 py-2 bg-gray-900 border border-gray-800 text-gray-400 text-xs rounded-lg disabled:opacity-40 hover:border-gray-700 transition-colors"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default AlertsPage;
