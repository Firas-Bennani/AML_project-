import { useState, useEffect } from 'react';
import { useAuthStore } from '../store/authStore';
import {
  getSummaryReport,
  getAnalystPerformance,
  getMissedFlags,
  getSARReports,
  getSARDetail,
  updateSARStatus,
  downloadSARPdf,
} from '../api/reportsApi';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

// ── Helpers ───────────────────────────────────────────────────
function formatDate(iso) {
  return new Date(iso).toLocaleDateString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function StatBlock({ label, value, sub, color = 'text-white' }) {
  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-5">
      <p className="text-gray-500 text-xs uppercase tracking-widest mb-2">{label}</p>
      <p className={`text-3xl font-bold ${color}`}>{value}</p>
      {sub && <p className="text-gray-600 text-xs mt-1">{sub}</p>}
    </div>
  );
}

function SectionHeader({ title, sub }) {
  return (
    <div className="mb-6">
      <h2 className="text-base font-semibold text-white">{title}</h2>
      {sub && <p className="text-gray-500 text-xs mt-1">{sub}</p>}
    </div>
  );
}

// ── Custom Tooltip ────────────────────────────────────────────
function CustomTooltip({ active, payload, label }) {
  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-sm">
        <p className="text-gray-400 mb-1">{label}</p>
        <p className="text-white font-semibold">{payload[0].value}</p>
      </div>
    );
  }
  return null;
}

// ── Tab 1 — Activity Summary ──────────────────────────────────
function SummaryTab() {
  const [data, setData] = useState(null);
  const [period, setPeriod] = useState('monthly');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getSummaryReport(period)
      .then(setData)
      .catch(() => setError('Failed to load summary report.'))
      .finally(() => setLoading(false));
  }, [period]);

  if (loading) return <Spinner />;
  if (error) return <ErrorBox message={error} />;
  if (!data) return null;

  const txnChartData = [
    { name: 'Total', value: data.transactions.total, color: '#3b82f6' },
    { name: 'Flagged', value: data.transactions.flagged, color: '#ef4444' },
    { name: 'Scored', value: data.transactions.scored_normal, color: '#22c55e' },
    { name: 'Reviewed', value: data.transactions.reviewed, color: '#a855f7' },
  ];
  // Note: "Scored" now counts every transaction with a non-NULL risk_score
  // (flagged + non-flagged), so the bar legitimately includes the flagged
  // bucket. This matches the user-facing definition fixed in bug #4.

  const alertChartData = [
    { name: 'Open', value: data.alerts.open, color: '#f59e0b' },
    { name: 'Under Review', value: data.alerts.under_review, color: '#3b82f6' },
    { name: 'Confirmed', value: data.alerts.confirmed, color: '#ef4444' },
    { name: 'Dismissed', value: data.alerts.dismissed, color: '#6b7280' },
  ];

  return (
    <div>
      {/* Period Selector */}
      <div className="flex items-center gap-3 mb-8">
        <span className="text-gray-500 text-xs">Period:</span>
        {['daily', 'weekly', 'monthly', 'all'].map((p) => (
          <button
            key={p}
            onClick={() => setPeriod(p)}
            className={`px-4 py-1.5 rounded-lg text-xs font-medium border transition-colors capitalize ${
              period === p
                ? 'bg-green-500/10 border-green-500/30 text-green-400'
                : 'bg-gray-900 border-gray-800 text-gray-400 hover:border-gray-700'
            }`}
          >
            {p === 'all' ? 'All Time' : p.charAt(0).toUpperCase() + p.slice(1)}
          </button>
        ))}
        <span className="text-gray-600 text-xs ml-auto">
          Generated {formatDate(data.generated_at)} · {data.period}
        </span>
      </div>

      {/* Transaction Stats */}
      <SectionHeader
        title="Transaction Activity"
        sub="Overview of transaction processing during the selected period"
      />
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatBlock label="Total" value={data.transactions.total} color="text-blue-400" />
        <StatBlock
          label="Flagged"
          value={data.transactions.flagged}
          sub={`${data.transactions.flag_rate_percent.toFixed(1)}% flag rate`}
          color="text-red-400"
        />
        <StatBlock
          label="Scored"
          value={data.transactions.scored_normal}
          sub="Transactions with a risk_score"
          color="text-green-400"
        />
        <StatBlock label="Reviewed" value={data.transactions.reviewed} color="text-purple-400" />
      </div>

      {/* Transaction Chart */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 mb-8">
        <h3 className="text-sm font-medium text-white mb-5">Transaction Breakdown</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={txnChartData} barSize={48}>
            <XAxis
              dataKey="name"
              tick={{ fill: '#6b7280', fontSize: 12 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: '#6b7280', fontSize: 12 }}
              axisLine={false}
              tickLine={false}
              allowDecimals={false}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {txnChartData.map((entry, i) => (
                <Cell key={i} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Alert Stats */}
      <SectionHeader
        title="Alert Activity"
        sub="Status breakdown of all alerts during the selected period"
      />
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatBlock label="Total Alerts" value={data.alerts.total} color="text-white" />
        <StatBlock label="Open" value={data.alerts.open} color="text-yellow-400" />
        <StatBlock
          label="Resolved"
          value={data.alerts.confirmed + data.alerts.dismissed}
          sub={`${data.alerts.resolution_rate_percent.toFixed(1)}% resolution rate`}
          color="text-green-400"
        />
        <StatBlock label="Under Review" value={data.alerts.under_review} color="text-blue-400" />
      </div>

      {/* Alert Chart */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 mb-8">
        <h3 className="text-sm font-medium text-white mb-5">Alert Status Breakdown</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={alertChartData} barSize={48}>
            <XAxis
              dataKey="name"
              tick={{ fill: '#6b7280', fontSize: 12 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: '#6b7280', fontSize: 12 }}
              axisLine={false}
              tickLine={false}
              allowDecimals={false}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {alertChartData.map((entry, i) => (
                <Cell key={i} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* User Activity */}
      <SectionHeader title="User Activity" sub="Login and session activity during the period" />
      <div className="grid grid-cols-3 gap-4">
        <StatBlock
          label="Active Users"
          value={data.user_activity.active_users}
          color="text-green-400"
        />
        <StatBlock
          label="Total Logins"
          value={data.user_activity.total_logins}
          color="text-blue-400"
        />
        <StatBlock
          label="Failed Logins"
          value={data.user_activity.failed_logins}
          color={data.user_activity.failed_logins > 0 ? 'text-red-400' : 'text-gray-400'}
        />
      </div>
    </div>
  );
}

// ── Tab 2 — Analyst Performance ───────────────────────────────
function AnalystTab() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getAnalystPerformance()
      .then(setData)
      .catch(() => setError('Failed to load analyst performance.'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Spinner />;
  if (error) return <ErrorBox message={error} />;
  if (!data) return null;

  const chartData = data.analysts.map((a) => ({
    name: a.analyst_name.split(' ')[0],
    Assigned: a.alerts_assigned,
    Confirmed: a.alerts_confirmed,
    Dismissed: a.alerts_dismissed,
  }));

  return (
    <div>
      <SectionHeader
        title="Analyst Performance"
        sub={`${data.total_analysts} analyst${data.total_analysts !== 1 ? 's' : ''} · Generated ${formatDate(data.generated_at)}`}
      />

      {/* Analyst Cards */}
      <div className="space-y-4 mb-8">
        {data.analysts.map((analyst) => (
          <div
            key={analyst.analyst_id}
            className={`bg-gray-900 border rounded-xl p-6 ${
              analyst.red_flag ? 'border-red-500/30' : 'border-gray-800'
            }`}
          >
            {/* Card Header */}
            <div className="flex items-start justify-between mb-5">
              <div>
                <div className="flex items-center gap-3">
                  <h3 className="text-white font-semibold">{analyst.analyst_name}</h3>
                  {analyst.red_flag && (
                    <span className="bg-red-500/10 border border-red-500/20 text-red-400 text-xs px-2 py-0.5 rounded-full">
                      ⚠ Red Flag
                    </span>
                  )}
                  {!analyst.is_active && (
                    <span className="bg-gray-500/10 border border-gray-500/20 text-gray-400 text-xs px-2 py-0.5 rounded-full">
                      Inactive
                    </span>
                  )}
                </div>
                <p className="text-gray-500 text-xs mt-1">{analyst.analyst_email}</p>
              </div>
              <div className="text-right">
                <p className="text-gray-500 text-xs">Dismissal Rate</p>
                <p
                  className={`text-xl font-bold mt-1 ${
                    analyst.dismissal_rate_percent > 60
                      ? 'text-red-400'
                      : analyst.dismissal_rate_percent > 30
                        ? 'text-yellow-400'
                        : 'text-green-400'
                  }`}
                >
                  {analyst.dismissal_rate_percent.toFixed(1)}%
                </p>
              </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-3 lg:grid-cols-6 gap-4">
              {[
                { label: 'Assigned', value: analyst.alerts_assigned, color: 'text-white' },
                { label: 'Confirmed', value: analyst.alerts_confirmed, color: 'text-red-400' },
                { label: 'Dismissed', value: analyst.alerts_dismissed, color: 'text-gray-400' },
                { label: 'In Review', value: analyst.alerts_under_review, color: 'text-blue-400' },
                {
                  label: 'Reviewed',
                  value: analyst.transactions_reviewed,
                  color: 'text-purple-400',
                },
                { label: 'Logins', value: analyst.total_logins, color: 'text-green-400' },
              ].map((s) => (
                <div key={s.label} className="bg-gray-800/50 rounded-lg p-3 text-center">
                  <p className="text-gray-500 text-xs mb-1">{s.label}</p>
                  <p className={`text-xl font-bold ${s.color}`}>{s.value}</p>
                </div>
              ))}
            </div>

            {/* Progress Bar — Alert Resolution */}
            {analyst.alerts_assigned > 0 && (
              <div className="mt-5">
                <div className="flex justify-between text-xs text-gray-500 mb-1.5">
                  <span>Alert Resolution Progress</span>
                  <span>
                    {analyst.alerts_confirmed + analyst.alerts_dismissed} /{' '}
                    {analyst.alerts_assigned} resolved
                  </span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-1.5">
                  <div
                    className="bg-green-500 h-1.5 rounded-full transition-all duration-500"
                    style={{
                      width: `${Math.min(
                        ((analyst.alerts_confirmed + analyst.alerts_dismissed) /
                          analyst.alerts_assigned) *
                          100,
                        100
                      )}%`,
                    }}
                  />
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Comparison Chart */}
      {chartData.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h3 className="text-sm font-medium text-white mb-5">Alert Handling Comparison</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData} barSize={24}>
              <XAxis
                dataKey="name"
                tick={{ fill: '#6b7280', fontSize: 12 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: '#6b7280', fontSize: 12 }}
                axisLine={false}
                tickLine={false}
                allowDecimals={false}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
              <Bar dataKey="Assigned" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Confirmed" fill="#ef4444" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Dismissed" fill="#6b7280" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

// ── Tab 3 — Missed Flags ──────────────────────────────────────
function MissedFlagsTab() {
  const [data, setData] = useState(null);
  const [threshold, setThreshold] = useState(0.6);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getMissedFlags(threshold)
      .then(setData)
      .catch(() => setError('Failed to load missed flags report.'))
      .finally(() => setLoading(false));
  }, [threshold]);

  if (loading) return <Spinner />;
  if (error) return <ErrorBox message={error} />;
  if (!data) return null;

  const getSeverityStyle = (severity) => {
    switch (severity) {
      case 'CRITICAL':
        return 'bg-red-500/10 text-red-400 border-red-500/20';
      case 'HIGH':
        return 'bg-orange-500/10 text-orange-400 border-orange-500/20';
      default:
        return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20';
    }
  };

  return (
    <div>
      <SectionHeader
        title="Missed Flags"
        sub="High-risk alerts that were dismissed — potential compliance gaps"
      />

      {/* Threshold Selector */}
      <div className="flex items-center gap-3 mb-8">
        <span className="text-gray-500 text-xs">Risk Threshold:</span>
        {[0.5, 0.6, 0.7, 0.8].map((t) => (
          <button
            key={t}
            onClick={() => setThreshold(t)}
            className={`px-4 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              threshold === t
                ? 'bg-green-500/10 border-green-500/30 text-green-400'
                : 'bg-gray-900 border-gray-800 text-gray-400 hover:border-gray-700'
            }`}
          >
            {(t * 100).toFixed(0)}%+
          </button>
        ))}
      </div>

      {/* Summary Counts */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatBlock label="Total Missed" value={data.total_missed_flags} color="text-white" />
        <StatBlock label="Critical" value={data.critical_count} color="text-red-400" />
        <StatBlock label="High" value={data.high_count} color="text-orange-400" />
        <StatBlock label="Medium" value={data.medium_count} color="text-yellow-400" />
      </div>

      {/* Missed Flag List */}
      {data.missed_flags.length === 0 ? (
        <div className="text-center py-16 border border-gray-800 rounded-xl">
          <p className="text-green-400 text-sm font-medium mb-1">No missed flags found</p>
          <p className="text-gray-600 text-xs">
            No dismissed alerts above the {(threshold * 100).toFixed(0)}% risk threshold.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {data.missed_flags.map((flag) => (
            <div key={flag.alert_id} className="bg-gray-900 border border-gray-800 rounded-xl p-5">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <p className="text-white text-sm font-medium">{flag.alert_reason}</p>
                  <p className="text-gray-500 text-xs mt-1">
                    {flag.sender_name} → {flag.receiver_name}
                  </p>
                </div>
                <span
                  className={`border rounded-full px-3 py-1 text-xs font-medium shrink-0 ml-4 ${getSeverityStyle(flag.severity)}`}
                >
                  {flag.severity}
                </span>
              </div>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
                <div>
                  <p className="text-gray-500 mb-0.5">Risk Score</p>
                  <p className="text-red-400 font-semibold">
                    {(flag.risk_score * 100).toFixed(0)}%
                  </p>
                </div>
                <div>
                  <p className="text-gray-500 mb-0.5">Amount</p>
                  <p className="text-white">
                    {flag.amount
                      ? new Intl.NumberFormat('en-US', { minimumFractionDigits: 2 }).format(
                          flag.amount
                        ) +
                        ' ' +
                        flag.currency
                      : 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-gray-500 mb-0.5">Dismissed By</p>
                  <p className="text-gray-300">{flag.dismissed_by || 'Unknown'}</p>
                </div>
                <div>
                  <p className="text-gray-500 mb-0.5">Resolved At</p>
                  <p className="text-gray-300">{formatDate(flag.resolved_at)}</p>
                </div>
              </div>
              {flag.alert_notes && (
                <div className="mt-3 pt-3 border-t border-gray-800">
                  <p className="text-gray-500 text-xs mb-1">Notes</p>
                  <p className="text-gray-400 text-xs">{flag.alert_notes}</p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Tab 4 — SAR Reports (Bug #4) ──────────────────────────────
function SARTab() {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [busyId, setBusyId] = useState(null);
  const pageSize = 10;

  const fetchList = async () => {
    setLoading(true);
    setError(null);
    try {
      const params = { skip: page * pageSize, limit: pageSize };
      if (statusFilter !== 'ALL') params.status = statusFilter;
      if (search.trim()) params.search = search.trim();
      const data = await getSARReports(params);
      setItems(data.items);
      setTotal(data.total);
    } catch {
      setError('Failed to load SAR reports.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchList();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, statusFilter]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    setPage(0);
    fetchList();
  };

  const handleView = async (alertId) => {
    setDetailLoading(true);
    try {
      const data = await getSARDetail(alertId);
      setDetail(data);
    } catch {
      setError('Failed to load SAR detail.');
    } finally {
      setDetailLoading(false);
    }
  };

  const handleMarkSubmitted = async (alertId) => {
    setBusyId(alertId);
    try {
      await updateSARStatus(alertId, 'SUBMITTED');
      await fetchList();
      if (detail && detail.alert_id === alertId) {
        const fresh = await getSARDetail(alertId);
        setDetail(fresh);
      }
    } catch {
      setError('Failed to update SAR status.');
    } finally {
      setBusyId(null);
    }
  };

  const handleDownload = async (alertId) => {
    setBusyId(alertId);
    try {
      await downloadSARPdf(alertId);
    } catch {
      setError('Failed to download PDF.');
    } finally {
      setBusyId(null);
    }
  };

  const statusStyle = (s) => {
    switch (s) {
      case 'DRAFT':
        return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20';
      case 'VALIDATED':
        return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
      case 'SUBMITTED':
        return 'bg-green-500/10 text-green-400 border-green-500/20';
      default:
        return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
    }
  };

  const tabs = ['ALL', 'DRAFT', 'VALIDATED', 'SUBMITTED'];
  const totalPages = Math.ceil(total / pageSize);

  return (
    <div>
      <SectionHeader
        title="SAR Reports"
        sub="Suspicious Activity Reports drafted by the investigation agent. Download as PDF or mark submitted to the FIU once filed."
      />

      <div className="flex flex-wrap items-center gap-3 mb-6">
        {tabs.map((t) => (
          <button
            key={t}
            onClick={() => {
              setPage(0);
              setStatusFilter(t);
            }}
            className={`px-4 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              statusFilter === t
                ? 'bg-green-500/10 border-green-500/30 text-green-400'
                : 'bg-gray-900 border-gray-800 text-gray-400 hover:border-gray-700'
            }`}
          >
            {t === 'ALL' ? 'All' : t.charAt(0) + t.slice(1).toLowerCase()}
          </button>
        ))}

        <form onSubmit={handleSearchSubmit} className="ml-auto flex items-center gap-2">
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search sender / receiver"
            className="bg-gray-900 border border-gray-800 text-white text-xs rounded-lg px-3 py-1.5 w-64 focus:outline-none focus:border-green-500"
          />
          <button
            type="submit"
            className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 border border-gray-700 text-xs text-gray-200 rounded-lg"
          >
            Search
          </button>
        </form>
      </div>

      {error && <ErrorBox message={error} />}
      {loading ? (
        <Spinner />
      ) : items.length === 0 ? (
        <div className="text-center py-16 border border-gray-800 rounded-xl">
          <p className="text-gray-500 text-sm">No SAR reports for this filter.</p>
        </div>
      ) : (
        <div className="overflow-x-auto border border-gray-800 rounded-xl">
          <table className="min-w-full text-xs">
            <thead className="bg-gray-900 text-gray-500 uppercase tracking-widest">
              <tr>
                <th className="text-left px-4 py-3">Generated</th>
                <th className="text-left px-4 py-3">Tx ID</th>
                <th className="text-left px-4 py-3">Sender → Receiver</th>
                <th className="text-left px-4 py-3">Amount</th>
                <th className="text-left px-4 py-3">Risk</th>
                <th className="text-left px-4 py-3">Status</th>
                <th className="text-right px-4 py-3">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {items.map((sar) => (
                <tr key={sar.alert_id} className="text-gray-300 hover:bg-gray-900/40">
                  <td className="px-4 py-3 whitespace-nowrap">
                    {sar.generated_at ? formatDate(sar.generated_at) : '—'}
                  </td>
                  <td className="px-4 py-3 font-mono text-gray-500">
                    {sar.transaction_id.slice(0, 8)}…
                  </td>
                  <td className="px-4 py-3">
                    <p className="text-gray-200">{sar.sender_name || '—'}</p>
                    <p className="text-gray-500">→ {sar.receiver_name || '—'}</p>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    {sar.amount != null
                      ? `${sar.amount.toLocaleString()} ${sar.currency || ''}`
                      : '—'}
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-red-400 font-semibold">
                      {(sar.risk_score * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`border rounded-full px-2 py-0.5 text-xs font-medium ${statusStyle(
                        sar.sar_status
                      )}`}
                    >
                      {sar.sar_status}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right whitespace-nowrap">
                    <div className="flex justify-end gap-2">
                      <button
                        onClick={() => handleView(sar.alert_id)}
                        className="px-2 py-1 border border-gray-700 hover:border-gray-500 rounded-md text-gray-300"
                      >
                        View
                      </button>
                      <button
                        onClick={() => handleDownload(sar.alert_id)}
                        disabled={busyId === sar.alert_id}
                        className="px-2 py-1 border border-gray-700 hover:border-gray-500 rounded-md text-gray-300 disabled:opacity-40"
                      >
                        PDF
                      </button>
                      <button
                        onClick={() => handleMarkSubmitted(sar.alert_id)}
                        disabled={
                          busyId === sar.alert_id || sar.sar_status === 'SUBMITTED'
                        }
                        className="px-2 py-1 bg-green-500/10 border border-green-500/30 hover:bg-green-500/20 rounded-md text-green-400 disabled:opacity-40"
                      >
                        Mark Submitted
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-6">
          <p className="text-gray-600 text-xs">
            Page {page + 1} of {totalPages} · {total} total
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

      {detail && (
        <div
          className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4"
          onClick={() => setDetail(null)}
        >
          <div
            className="bg-gray-950 border border-gray-800 rounded-2xl max-w-3xl w-full max-h-[85vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="sticky top-0 bg-gray-950 border-b border-gray-800 px-6 py-4 flex items-center justify-between">
              <div>
                <h3 className="text-white font-semibold">SAR Detail</h3>
                <p className="text-gray-500 text-xs">{detail.alert_id}</p>
              </div>
              <button
                onClick={() => setDetail(null)}
                className="text-gray-400 hover:text-white text-xl"
              >
                ×
              </button>
            </div>
            {detailLoading ? (
              <Spinner />
            ) : (
              <div className="px-6 py-5 space-y-5 text-sm">
                <section>
                  <h4 className="text-xs uppercase text-gray-500 tracking-widest mb-2">
                    Reporting Institution
                  </h4>
                  <p className="text-gray-300">{detail.reporting_institution}</p>
                  <p className="text-gray-500 text-xs">{detail.jurisdiction}</p>
                </section>
                <section>
                  <h4 className="text-xs uppercase text-gray-500 tracking-widest mb-2">
                    Suspect
                  </h4>
                  <p className="text-gray-200">
                    {detail.suspect_name || '—'}{' '}
                    <span className="text-gray-500 text-xs">
                      ({detail.suspect_account || '—'})
                    </span>
                  </p>
                  <p className="text-gray-500 text-xs">
                    Counterparty: {detail.counterparty_name || '—'} (
                    {detail.counterparty_account || '—'})
                  </p>
                </section>
                <section>
                  <h4 className="text-xs uppercase text-gray-500 tracking-widest mb-2">
                    Suspicious Activity
                  </h4>
                  <p className="text-gray-300">
                    {detail.amount != null
                      ? `${detail.amount.toLocaleString()} ${detail.currency || ''}`
                      : '—'}{' '}
                    · {detail.transaction_type || '—'}
                  </p>
                  <p className="text-gray-500 text-xs">
                    Risk score: {(detail.risk_score * 100).toFixed(0)}% · Typologies:{' '}
                    {detail.typologies?.join(', ') || '—'}
                  </p>
                  <p className="text-gray-500 text-xs">
                    Rule hits: {detail.rule_hits?.join(', ') || '—'}
                  </p>
                </section>
                <section>
                  <h4 className="text-xs uppercase text-gray-500 tracking-widest mb-2">
                    Narrative (EN)
                  </h4>
                  <pre className="bg-gray-900 border border-gray-800 rounded-lg p-3 text-gray-300 whitespace-pre-wrap text-xs">
                    {detail.sar_en || '(not yet generated)'}
                  </pre>
                </section>
                <section>
                  <h4 className="text-xs uppercase text-gray-500 tracking-widest mb-2">
                    Narrative (FR)
                  </h4>
                  <pre className="bg-gray-900 border border-gray-800 rounded-lg p-3 text-gray-300 whitespace-pre-wrap text-xs">
                    {detail.sar_fr || '(non encore généré)'}
                  </pre>
                </section>
                <section className="flex gap-3">
                  <button
                    onClick={() => handleDownload(detail.alert_id)}
                    className="px-4 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 text-xs rounded-lg text-gray-200"
                  >
                    Download PDF
                  </button>
                  <button
                    onClick={() => handleMarkSubmitted(detail.alert_id)}
                    disabled={detail.sar_status === 'SUBMITTED'}
                    className="px-4 py-2 bg-green-500/10 border border-green-500/30 hover:bg-green-500/20 rounded-lg text-xs text-green-400 disabled:opacity-40"
                  >
                    Mark Submitted
                  </button>
                </section>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Shared UI ─────────────────────────────────────────────────
function Spinner() {
  return (
    <div className="flex items-center justify-center py-20">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-green-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-gray-500 text-sm">Loading...</p>
      </div>
    </div>
  );
}

function ErrorBox({ message }) {
  return (
    <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-6 text-center">
      <p className="text-red-400 text-sm">{message}</p>
    </div>
  );
}

// ── Main Reports Page ─────────────────────────────────────────
function ReportsPage() {
  const { user } = useAuthStore();
  const canAccess = user?.role === 'ADMIN' || user?.role === 'AUDITOR';
  const [activeTab, setActiveTab] = useState('summary');

  // Access denied for analysts
  if (!canAccess) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-10 text-center max-w-md">
          <p className="text-3xl mb-4">🔒</p>
          <h2 className="text-white font-semibold text-lg mb-2">Access Restricted</h2>
          <p className="text-gray-500 text-sm">
            Reports are only available to Admin and Auditor roles.
          </p>
        </div>
      </div>
    );
  }

  const tabs = [
    { key: 'summary', label: 'Activity Summary' },
    { key: 'analysts', label: 'Analyst Performance' },
    { key: 'missed', label: 'Missed Flags' },
    { key: 'sar', label: 'SAR Reports' },
  ];

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <div className="max-w-6xl mx-auto px-6 py-10">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-white mb-1">Reports</h1>
          <p className="text-gray-500 text-sm">Compliance reporting and performance analytics</p>
        </div>

        {/* Tab Bar */}
        <div className="flex gap-2 mb-8 border-b border-gray-800 pb-0">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`px-5 py-3 text-sm font-medium border-b-2 transition-colors -mb-px ${
                activeTab === tab.key
                  ? 'border-green-500 text-green-400'
                  : 'border-transparent text-gray-400 hover:text-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {activeTab === 'summary' && <SummaryTab />}
        {activeTab === 'analysts' && <AnalystTab />}
        {activeTab === 'missed' && <MissedFlagsTab />}
        {activeTab === 'sar' && <SARTab />}
      </div>
    </div>
  );
}

export default ReportsPage;
