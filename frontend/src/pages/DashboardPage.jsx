import { useState, useEffect } from 'react';
import { useAuthStore } from '../store/authStore';
import { getTransactions } from '../api/transactionsApi';
import { getAlerts } from '../api/alertsApi';
import { getSummaryReport } from '../api/reportsApi';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

// ── Stat Card Component ───────────────────────────────────────
function StatCard({ label, value, sub, color }) {
  const colors = {
    green: 'border-green-500/30 bg-green-500/5',
    yellow: 'border-yellow-500/30 bg-yellow-500/5',
    red: 'border-red-500/30 bg-red-500/5',
    blue: 'border-blue-500/30 bg-blue-500/5',
  };
  const textColors = {
    green: 'text-green-400',
    yellow: 'text-yellow-400',
    red: 'text-red-400',
    blue: 'text-blue-400',
  };

  return (
    <div className={`border rounded-xl p-6 ${colors[color]}`}>
      <p className="text-gray-500 text-xs font-medium uppercase tracking-widest mb-3">{label}</p>
      <p className={`text-4xl font-bold mb-1 ${textColors[color]}`}>{value}</p>
      {sub && <p className="text-gray-600 text-xs mt-2">{sub}</p>}
    </div>
  );
}

// ── Custom Tooltip for Chart ──────────────────────────────────
function CustomTooltip({ active, payload, label }) {
  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-sm">
        <p className="text-gray-400 mb-1">{label}</p>
        <p className="text-white font-semibold">{payload[0].value} transactions</p>
      </div>
    );
  }
  return null;
}

// ── Main Dashboard ────────────────────────────────────────────
function DashboardPage() {
  const { user } = useAuthStore();
  const isPrivileged = user?.role === 'ADMIN' || user?.role === 'AUDITOR';

  const [stats, setStats] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [alertStats, setAlertStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);

        // Fetch transactions and alerts — available to ALL roles
        const [txnData, alertData] = await Promise.all([
          getTransactions({ limit: 100 }),
          getAlerts({ limit: 100 }),
        ]);

        // Count transaction types for the bar chart
        const typeCounts = { DEPOSIT: 0, WITHDRAWAL: 0, TRANSFER: 0 };
        txnData.items.forEach((t) => {
          if (typeCounts[t.type] !== undefined) typeCounts[t.type]++;
        });

        setChartData([
          { name: 'Deposits', count: typeCounts.DEPOSIT, color: '#22c55e' },
          { name: 'Withdrawals', count: typeCounts.WITHDRAWAL, color: '#f59e0b' },
          { name: 'Transfers', count: typeCounts.TRANSFER, color: '#3b82f6' },
        ]);

        // Count alert statuses
        const alertCounts = { OPEN: 0, UNDER_REVIEW: 0, CONFIRMED: 0, DISMISSED: 0 };
        alertData.items.forEach((a) => {
          if (alertCounts[a.status] !== undefined) alertCounts[a.status]++;
        });
        setAlertStats(alertCounts);

        // If admin or auditor, also fetch the summary report
        if (isPrivileged) {
          const summary = await getSummaryReport('weekly');
          setStats(summary);
        } else {
          // Build a minimal stats object from transactions and alerts
          const flagged = txnData.items.filter((t) => t.status === 'FLAGGED').length;
          setStats({
            transactions: {
              total: txnData.total,
              flagged,
              flag_rate_percent: txnData.total ? ((flagged / txnData.total) * 100).toFixed(1) : 0,
            },
            alerts: {
              total: alertData.total,
              open: alertCounts.OPEN,
              confirmed: alertCounts.CONFIRMED,
              dismissed: alertCounts.DISMISSED,
              resolution_rate_percent:
                alertData.total > 0
                  ? (
                      ((alertCounts.CONFIRMED + alertCounts.DISMISSED) / alertData.total) *
                      100
                    ).toFixed(1)
                  : 0,
            },
          });
        }
      } catch (err) {
        setError('Failed to load dashboard data. Please try again.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, [isPrivileged]);

  // ── Loading State ───────────────────────────────────────────
  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-green-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-500 text-sm">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  // ── Error State ─────────────────────────────────────────────
  if (error) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-8 text-center max-w-md">
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      </div>
    );
  }

  // ── Dashboard ───────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <div className="max-w-7xl mx-auto px-6 py-10">
        {/* Header */}
        <div className="mb-10 flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white mb-1">Dashboard</h1>
            <p className="text-gray-500 text-sm">
              Welcome back, <span className="text-gray-300">{user?.name}</span> ·{' '}
              <span className="text-green-500 text-xs font-medium uppercase tracking-wider">
                {user?.role}
              </span>
            </p>
          </div>
          <div className="text-right">
            <p className="text-gray-600 text-xs">
              {isPrivileged ? 'Weekly Summary' : 'Live Overview'}
            </p>
            <p className="text-gray-500 text-xs mt-1">
              {new Date().toLocaleDateString('en-GB', {
                weekday: 'long',
                year: 'numeric',
                month: 'long',
                day: 'numeric',
              })}
            </p>
          </div>
        </div>

        {/* Stat Cards */}
        {stats && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
            <StatCard
              label="Total Transactions"
              value={stats.transactions.total}
              sub={isPrivileged ? 'Last 7 days' : 'All transactions'}
              color="green"
            />
            <StatCard
              label="Flagged Transactions"
              value={stats.transactions.flagged}
              sub={`${Number(stats.transactions.flag_rate_percent).toFixed(1)}% flag rate`}
              color="red"
            />
            <StatCard
              label="Open Alerts"
              value={stats.alerts.open}
              sub={`${stats.alerts.total} total alerts`}
              color="yellow"
            />
            <StatCard
              label="Resolution Rate"
              value={`${Number(stats.alerts.resolution_rate_percent).toFixed(1)}%`}
              sub={`${stats.alerts.confirmed} confirmed · ${stats.alerts.dismissed} dismissed`}
              color="blue"
            />
          </div>
        )}

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Bar Chart — Transactions by Type */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h2 className="text-sm font-semibold text-white mb-1">Transactions by Type</h2>
            <p className="text-gray-600 text-xs mb-6">
              Distribution across deposit, withdrawal, and transfer
            </p>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={chartData} barSize={40}>
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
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={index} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Alert Status Breakdown */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h2 className="text-sm font-semibold text-white mb-1">Alert Status Breakdown</h2>
            <p className="text-gray-600 text-xs mb-6">Current state of all alerts in the system</p>
            {alertStats && (
              <div className="space-y-4 mt-2">
                {[
                  {
                    label: 'Open',
                    value: alertStats.OPEN,
                    color: 'bg-yellow-500',
                    text: 'text-yellow-400',
                  },
                  {
                    label: 'Under Review',
                    value: alertStats.UNDER_REVIEW,
                    color: 'bg-blue-500',
                    text: 'text-blue-400',
                  },
                  {
                    label: 'Confirmed',
                    value: alertStats.CONFIRMED,
                    color: 'bg-red-500',
                    text: 'text-red-400',
                  },
                  {
                    label: 'Dismissed',
                    value: alertStats.DISMISSED,
                    color: 'bg-gray-500',
                    text: 'text-gray-400',
                  },
                ].map((item) => {
                  const total =
                    alertStats.OPEN +
                    alertStats.UNDER_REVIEW +
                    alertStats.CONFIRMED +
                    alertStats.DISMISSED;
                  const pct = total > 0 ? (item.value / total) * 100 : 0;
                  return (
                    <div key={item.label}>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-gray-400 text-xs">{item.label}</span>
                        <span className={`text-xs font-semibold ${item.text}`}>{item.value}</span>
                      </div>
                      <div className="w-full bg-gray-800 rounded-full h-1.5">
                        <div
                          className={`${item.color} h-1.5 rounded-full transition-all duration-500`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default DashboardPage;
