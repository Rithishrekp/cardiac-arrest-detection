import { useEffect, useState } from 'react';
import { api } from '../services/api';
import RiskCard from '../components/RiskCard';

const RISK_EMOJI = { normal: '✅', medium: '⚠️', critical: '🚨' };

export default function History({ onNavigate }) {
  const [records, setRecords] = useState([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    loadHistory('');
  }, []);

  async function loadHistory(query) {
    setLoading(true);
    setError(null);
    try {
      const data = await api.history(query);
      setRecords(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleSearch(e) {
    e.preventDefault();
    loadHistory(search);
  }

  async function viewDetail(id) {
    setDetailLoading(true);
    try {
      const data = await api.getRecord(id);
      setSelected(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setDetailLoading(false);
    }
  }

  if (selected) {
    return (
      <div className="animate-fadeIn">
        <div className="page-header">
          <button className="btn btn-ghost btn-sm" onClick={() => setSelected(null)}>
            ← Back to History
          </button>
          <h1 className="page-title">Record #{selected.record.id}</h1>
        </div>
        <div className="page-body">
          {detailLoading ? (
            <div className="loading-overlay">
              <div className="spinner" />
              <span>Loading record…</span>
            </div>
          ) : (
            <RiskCard
              record={selected.record}
              suggestions={selected.suggestions}
              disclaimer={selected.disclaimer}
            />
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="animate-fadeIn">
      <div className="page-header">
        <h1 className="page-title">Prediction History</h1>
        <p className="page-subtitle">All previous cardiac risk assessments</p>
      </div>

      <div className="page-body">
        <div className="history-toolbar card">
          <form onSubmit={handleSearch} className="search-form">
            <input
              type="text"
              placeholder="Search by patient name or ID…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <button type="submit" className="btn btn-primary btn-sm">
              Search
            </button>
            {search && (
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => {
                  setSearch('');
                  loadHistory('');
                }}
              >
                Clear
              </button>
            )}
          </form>
          <button className="btn btn-primary btn-sm" onClick={() => onNavigate('predict')}>
            + New Assessment
          </button>
        </div>

        {loading ? (
          <div className="loading-overlay">
            <div className="spinner" />
            <span>Loading history…</span>
          </div>
        ) : error ? (
          <div className="alert danger">
            <span className="alert-icon">⚠️</span>
            <div>
              <strong>Failed to load history</strong>
              <br />
              {error}
            </div>
          </div>
        ) : records.length === 0 ? (
          <div className="empty-state card">
            <span className="empty-icon">📭</span>
            <h3>No records found</h3>
            <p>Submit your first cardiac risk prediction to see it here.</p>
            <button className="btn btn-primary" onClick={() => onNavigate('predict')}>
              Start Assessment →
            </button>
          </div>
        ) : (
          <div className="card">
            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Patient</th>
                    <th>Demographics</th>
                    <th>ECG Intervals (ms)</th>
                    <th>Risk</th>
                    <th>Score</th>
                    <th>Date</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {records.map((rec) => (
                    <tr key={rec.id}>
                      <td className="mono">#{rec.id}</td>
                      <td>
                        <div className="cell-primary">{rec.patient_name}</div>
                        <div className="cell-muted">{rec.patient_id}</div>
                      </td>
                      <td className="cell-muted">
                        {rec.age}y · {rec.gender}
                      </td>
                      <td className="cell-muted mono">
                        RR {rec.rr_interval} · PP {rec.pp_interval} · QT {rec.qt_interval}
                      </td>
                      <td>
                        <span className={`risk-badge ${rec.risk_level}`}>
                          {RISK_EMOJI[rec.risk_level]} {rec.risk_level}
                        </span>
                      </td>
                      <td className={`score-cell ${rec.risk_level}`}>
                        {Math.round(rec.risk_score * 10) / 10}%
                      </td>
                      <td className="cell-muted">
                        {new Date(rec.created_at).toLocaleString()}
                      </td>
                      <td>
                        <button
                          className="btn btn-ghost btn-sm"
                          onClick={() => viewDetail(rec.id)}
                        >
                          View →
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
