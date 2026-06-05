import { useState, useEffect, useCallback } from 'react';
import { getTransactions } from '../api/transactionsApi';

// ── Helpers ───────────────────────────────────────────────────
// Per-typology thresholds — must mirror the AI checkpoint's calibrated values
// (currently [smurf=0.75, struct=0.80, layer=0.75]). Above threshold = flagged.
const TYP_THRESHOLDS = { smurfing: 0.75, structuring: 0.80, layering: 0.75 };

function getTypologyBadge(score, threshold) {
  if (score === null || score === undefined) {
    return { label: '—', classes: 'bg-gray-800 text-gray-600' };
  }
  const above = score >= threshold;
  const label = `${(score * 100).toFixed(0)}%`;
  return above
    ? { label, classes: 'bg-red-500/10 text-red-400 border border-red-500/20' }
    : { label, classes: 'bg-green-500/10 text-green-400 border border-green-500/20' };
}

// 3-tier traffic light against the overall risk_score. Numeric label is
// shown to one decimal place. Old rows with no score get a neutral "—".
function getRiskBadge(score) {
  if (score === null || score === undefined) {
    return { label: '—', classes: 'bg-gray-800 text-gray-600' };
  }
  const label = `${(score * 100).toFixed(1)}%`;
  if (score < 0.40)
    return { label, classes: 'bg-green-500/10 text-green-400 border border-green-500/20' };
  if (score < 0.60)
    return { label, classes: 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/20' };
  return { label, classes: 'bg-red-500/10 text-red-400 border border-red-500/20' };
}

function getDisplayStatus(txn) {
  // WARMING_UP wins (account has < 3 prior txs → score not reliable).
  if (txn.status === 'WARMING_UP') return 'WARMING UP';
  // If we have typology data, derive from per-typology thresholds.
  const sm = txn.smurfing_score, st = txn.structuring_score, la = txn.layering_score;
  if (sm !== null && sm !== undefined) {
    const flagged =
      sm >= TYP_THRESHOLDS.smurfing
      || st >= TYP_THRESHOLDS.structuring
      || la >= TYP_THRESHOLDS.layering;
    return flagged ? 'FLAGGED' : 'CLEAN';
  }
  // Older rows with no typology — fall back to backend's coarse status.
  return txn.status === 'FLAGGED' ? 'FLAGGED' : 'CLEAN';
}

function getStatusStyle(displayStatus) {
  switch (displayStatus) {
    case 'FLAGGED':
      return 'bg-red-500/10 text-red-400 border border-red-500/20';
    case 'WARMING UP':
      return 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/20';
    case 'CLEAN':
      return 'bg-green-500/10 text-green-400 border border-green-500/20';
    default:
      return 'bg-gray-500/10 text-gray-400 border border-gray-500/20';
  }
}

function getTypeStyle(type) {
  switch (type) {
    case 'TRANSFER':
      return 'text-blue-400';
    case 'WITHDRAWAL':
      return 'text-orange-400';
    case 'DEPOSIT':
      return 'text-green-400';
    default:
      return 'text-gray-400';
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

function formatAmount(amount, currency) {
  return (
    new Intl.NumberFormat('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(amount) +
    ' ' +
    currency
  );
}

// ── Transaction Detail Drawer ─────────────────────────────────
function DetailDrawer({ transaction, onClose }) {
  if (!transaction) return null;
  const sm = getTypologyBadge(transaction.smurfing_score,    TYP_THRESHOLDS.smurfing);
  const st = getTypologyBadge(transaction.structuring_score, TYP_THRESHOLDS.structuring);
  const la = getTypologyBadge(transaction.layering_score,    TYP_THRESHOLDS.layering);
  const displayStatus = getDisplayStatus(transaction);

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      {/* Drawer */}
      <div className="relative w-full max-w-md bg-gray-900 border-l border-gray-800 h-full overflow-y-auto p-6 shadow-2xl">
        {/* Header */}
        <div className="flex items-start justify-between mb-6">
          <div>
            <h2 className="text-white font-semibold text-base">Transaction Detail</h2>
            <p className="text-gray-500 text-xs font-mono mt-1">{transaction.id}</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-white text-xl leading-none mt-1"
          >
            ✕
          </button>
        </div>

        {/* Amount + Status */}
        <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-5 mb-6 text-center">
          <p className="text-3xl font-bold text-white mb-1">
            {formatAmount(transaction.amount, transaction.currency)}
          </p>
          <p className={`text-sm font-medium ${getTypeStyle(transaction.type)}`}>
            {transaction.type}
          </p>
        </div>

        {/* Typology Badges + Status */}
        <div className="grid grid-cols-3 gap-2 mb-3">
          <div className={`rounded-lg px-2 py-2 text-center text-[10px] font-semibold ${sm.classes}`}>
            <div className="opacity-70">SMURF</div>
            <div className="text-sm mt-0.5">{sm.label}</div>
          </div>
          <div className={`rounded-lg px-2 py-2 text-center text-[10px] font-semibold ${st.classes}`}>
            <div className="opacity-70">STRUCT</div>
            <div className="text-sm mt-0.5">{st.label}</div>
          </div>
          <div className={`rounded-lg px-2 py-2 text-center text-[10px] font-semibold ${la.classes}`}>
            <div className="opacity-70">LAYER</div>
            <div className="text-sm mt-0.5">{la.label}</div>
          </div>
        </div>
        <div className={`rounded-lg px-3 py-2 text-center text-xs font-semibold mb-6 ${getStatusStyle(displayStatus)}`}>
          {displayStatus}
        </div>

        {/* Sender */}
        <div className="mb-5">
          <h3 className="text-xs font-medium text-gray-500 uppercase tracking-widest mb-3">
            Sender
          </h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-500 text-xs">Name</span>
              <span className="text-gray-300 text-xs font-medium">{transaction.sender_name}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500 text-xs">Account</span>
              <span className="text-gray-400 text-xs font-mono">{transaction.sender_account}</span>
            </div>
          </div>
        </div>

        <div className="border-t border-gray-800 my-4" />

        {/* Receiver */}
        <div className="mb-5">
          <h3 className="text-xs font-medium text-gray-500 uppercase tracking-widest mb-3">
            Receiver
          </h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-500 text-xs">Name</span>
              <span className="text-gray-300 text-xs font-medium">{transaction.receiver_name}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500 text-xs">Account</span>
              <span className="text-gray-400 text-xs font-mono">
                {transaction.receiver_account}
              </span>
            </div>
          </div>
        </div>

        <div className="border-t border-gray-800 my-4" />

        {/* Metadata */}
        <div className="mb-5">
          <h3 className="text-xs font-medium text-gray-500 uppercase tracking-widest mb-3">
            Metadata
          </h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-500 text-xs">Created</span>
              <span className="text-gray-400 text-xs">{formatDate(transaction.created_at)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500 text-xs">Reviewed By</span>
              <span className="text-gray-400 text-xs font-mono">
                {transaction.reviewed_by
                  ? transaction.reviewed_by.slice(0, 8) + '...'
                  : 'Not reviewed'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Main Transactions Page ────────────────────────────────────
function TransactionsPage() {
  const [transactions, setTransactions] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);

  // Filters
  const [statusFilter, setStatusFilter] = useState('');
  const [minAmount, setMinAmount] = useState('');
  const [maxAmount, setMaxAmount] = useState('');
  const [page, setPage] = useState(0);
  const pageSize = 15;

  const fetchTransactions = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = {
        skip: page * pageSize,
        limit: pageSize,
      };
      if (statusFilter) params.status = statusFilter;
      if (minAmount) params.min_amount = parseFloat(minAmount);
      if (maxAmount) params.max_amount = parseFloat(maxAmount);

      const data = await getTransactions(params);
      setTransactions(data.items);
      setTotal(data.total);
    } catch (err) {
      setError('Failed to load transactions. Please try again.');
    } finally {
      setLoading(false);
    }
  }, [page, statusFilter, minAmount, maxAmount]);

  useEffect(() => {
    fetchTransactions();
  }, [fetchTransactions]);

  // Reset to page 0 when filters change
  const handleFilterChange = (setter) => (e) => {
    setPage(0);
    setter(e.target.value);
  };

  const totalPages = Math.ceil(total / pageSize);

  // ── Loading ─────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-green-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-500 text-sm">Loading transactions...</p>
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
            onClick={fetchTransactions}
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
      <div className="max-w-7xl mx-auto px-6 py-10">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-white mb-1">Transactions</h1>
          <p className="text-gray-500 text-sm">
            {total} transaction{total !== 1 ? 's' : ''} · Click any row to view full details
          </p>
        </div>

        {/* Filters */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 mb-6 flex flex-wrap gap-4 items-end">
          {/* Status Filter */}
          <div>
            <label className="block text-xs text-gray-500 mb-1.5">Status</label>
            <select
              value={statusFilter}
              onChange={handleFilterChange(setStatusFilter)}
              className="bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:border-green-500 min-w-[140px]"
            >
              <option value="">All Statuses</option>
              <option value="FLAGGED">Flagged</option>
              <option value="SCORED">Scored</option>
            </select>
          </div>

          {/* Min Amount */}
          <div>
            <label className="block text-xs text-gray-500 mb-1.5">Min Amount</label>
            <input
              type="number"
              value={minAmount}
              onChange={handleFilterChange(setMinAmount)}
              placeholder="e.g. 10000"
              className="bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:border-green-500 w-36 placeholder-gray-600"
            />
          </div>

          {/* Max Amount */}
          <div>
            <label className="block text-xs text-gray-500 mb-1.5">Max Amount</label>
            <input
              type="number"
              value={maxAmount}
              onChange={handleFilterChange(setMaxAmount)}
              placeholder="e.g. 500000"
              className="bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:border-green-500 w-36 placeholder-gray-600"
            />
          </div>

          {/* Clear Filters */}
          {(statusFilter || minAmount || maxAmount) && (
            <button
              onClick={() => {
                setStatusFilter('');
                setMinAmount('');
                setMaxAmount('');
                setPage(0);
              }}
              className="text-xs text-gray-400 hover:text-white underline pb-2"
            >
              Clear filters
            </button>
          )}
        </div>

        {/* Table */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
          {/* Table Header — 15-col arbitrary grid (Tailwind v4 native named
              steps stop at 12; arbitrary-value syntax is guaranteed to emit).
              Layout: 2 sender / 2 receiver / 2 amount / 1 risk / 1 type /
                      3 typology / 2 status / 2 date = 15. */}
          <div className="grid grid-cols-[repeat(15,minmax(0,1fr))] gap-4 px-5 py-3 border-b border-gray-800">
            <div className="col-span-2 text-xs font-medium text-gray-500 uppercase tracking-widest">Sender</div>
            <div className="col-span-2 text-xs font-medium text-gray-500 uppercase tracking-widest">Receiver</div>
            <div className="col-span-2 text-xs font-medium text-gray-500 uppercase tracking-widest">Amount</div>
            <div className="col-span-1 text-xs font-medium text-gray-500 uppercase tracking-widest" title="Overall risk score">Risk %</div>
            <div className="col-span-1 text-xs font-medium text-gray-500 uppercase tracking-widest">Type</div>
            <div className="col-span-1 text-xs font-medium text-gray-500 uppercase tracking-widest" title="Smurfing typology">Smurf</div>
            <div className="col-span-1 text-xs font-medium text-gray-500 uppercase tracking-widest" title="Structuring typology">Struct</div>
            <div className="col-span-1 text-xs font-medium text-gray-500 uppercase tracking-widest" title="Layering typology">Layer</div>
            <div className="col-span-2 text-xs font-medium text-gray-500 uppercase tracking-widest">Status</div>
            <div className="col-span-2 text-xs font-medium text-gray-500 uppercase tracking-widest">Date</div>
          </div>

          {/* Table Rows */}
          {transactions.length === 0 ? (
            <div className="text-center py-16">
              <p className="text-gray-500 text-sm">No transactions match your filters.</p>
            </div>
          ) : (
            transactions.map((txn) => {
              const risk = getRiskBadge(txn.risk_score);
              const sm = getTypologyBadge(txn.smurfing_score,    TYP_THRESHOLDS.smurfing);
              const st = getTypologyBadge(txn.structuring_score, TYP_THRESHOLDS.structuring);
              const la = getTypologyBadge(txn.layering_score,    TYP_THRESHOLDS.layering);
              const displayStatus = getDisplayStatus(txn);
              return (
                <div
                  key={txn.id}
                  onClick={() => setSelected(txn)}
                  className="grid grid-cols-[repeat(15,minmax(0,1fr))] gap-4 px-5 py-4 border-b border-gray-800/50 hover:bg-gray-800/30 cursor-pointer transition-colors last:border-b-0"
                >
                  {/* Sender */}
                  <div className="col-span-2 min-w-0">
                    <p className="text-white text-sm truncate">{txn.sender_name}</p>
                    <p className="text-gray-600 text-xs font-mono truncate">
                      {txn.sender_account.slice(0, 10)}...
                    </p>
                  </div>
                  {/* Receiver */}
                  <div className="col-span-2 min-w-0">
                    <p className="text-white text-sm truncate">{txn.receiver_name}</p>
                    <p className="text-gray-600 text-xs font-mono truncate">
                      {txn.receiver_account.slice(0, 10)}...
                    </p>
                  </div>
                  {/* Amount */}
                  <div className="col-span-2">
                    <p className="text-white text-sm font-medium">
                      {formatAmount(txn.amount, txn.currency)}
                    </p>
                  </div>
                  {/* Risk % */}
                  <div className="col-span-1">
                    <span className={`text-xs font-semibold px-2 py-1 rounded-md ${risk.classes}`}>{risk.label}</span>
                  </div>
                  {/* Type */}
                  <div className="col-span-1">
                    <span className={`text-xs font-medium ${getTypeStyle(txn.type)}`}>
                      {txn.type}
                    </span>
                  </div>
                  {/* Smurfing */}
                  <div className="col-span-1">
                    <span className={`text-xs font-semibold px-2 py-1 rounded-md ${sm.classes}`}>{sm.label}</span>
                  </div>
                  {/* Structuring */}
                  <div className="col-span-1">
                    <span className={`text-xs font-semibold px-2 py-1 rounded-md ${st.classes}`}>{st.label}</span>
                  </div>
                  {/* Layering */}
                  <div className="col-span-1">
                    <span className={`text-xs font-semibold px-2 py-1 rounded-md ${la.classes}`}>{la.label}</span>
                  </div>
                  {/* Status */}
                  <div className="col-span-2">
                    <span className={`text-xs px-2 py-1 rounded-md font-medium ${getStatusStyle(displayStatus)}`}>
                      {displayStatus}
                    </span>
                  </div>
                  {/* Date */}
                  <div className="col-span-2">
                    <p className="text-gray-400 text-xs">{formatDate(txn.created_at)}</p>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between mt-6">
            <p className="text-gray-600 text-xs">
              Showing {page * pageSize + 1}–{Math.min((page + 1) * pageSize, total)} of {total}{' '}
              transactions
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="px-4 py-2 bg-gray-900 border border-gray-800 text-gray-400 text-xs rounded-lg disabled:opacity-40 hover:border-gray-700 transition-colors"
              >
                Previous
              </button>
              <span className="px-4 py-2 text-gray-500 text-xs">
                {page + 1} / {totalPages}
              </span>
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

      {/* Detail Drawer */}
      <DetailDrawer transaction={selected} onClose={() => setSelected(null)} />
    </div>
  );
}

export default TransactionsPage;
