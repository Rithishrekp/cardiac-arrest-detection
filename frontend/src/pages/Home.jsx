import React, { useEffect, useState } from 'react';
import { api } from '../services/api';

const RISK_COLORS = {
  normal:   'var(--risk-normal)',
  medium:   'var(--risk-medium)',
  critical: 'var(--risk-critical)',
};

function RiskBar({ label, count, total, color }) {
  const pct = total > 0 ? Math.round((count / total) * 100) : 0;
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 13 }}>
        <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
        <span style={{ color, fontWeight: 700 }}>{count} <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>({pct}%)</span></span>
      </div>
      <div style={{ height: 7, background: 'var(--border)', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 4, transition: 'width 1s ease' }} />
      </div>
    </div>
  );
}

function RecentRow({ rec, onClick }) {
  const riskColor = RISK_COLORS[rec.risk_level] || 'var(--text-secondary)';
  const date = new Date(rec.created_at).toLocaleString('en-IN', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' });
  return (
    <tr style={{ cursor: 'pointer' }} onClick={() => onClick(rec)}>
      <td>
        <div style={{ fontWeight: 600, fontSize: 14 }}>{rec.patient_name}</div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{rec.patient_id}</div>
      </td>
      <td style={{ color: 'var(--text-secondary)', fontSize: 13 }}>{rec.age}y · {rec.gender}</td>
      <td>
        <span className={`risk-badge ${rec.risk_level}`}>
          {rec.risk_level === 'normal' ? '✅' : rec.risk_level === 'medium' ? '⚠️' : '🚨'} {rec.risk_label}
        </span>
      </td>
      <td style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, color: riskColor, fontSize: 15 }}>
        {rec.risk_score}%
      </td>
      <td style={{ color: 'var(--text-muted)', fontSize: 12 }}>{date}</td>
    </tr>
  );
}

export default function Home({ onNavigate }) {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.stats()
      .then(setStats)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div>
      <div className="page-header"><h1 className="page-title">Dashboard</h1></div>
      <div className="page-body">
        <div className="loading-overlay"><div className="spinner" /><span>Loading dashboard…</span></div>
      </div>
    </div>
  );

  if (error) return (
    <div>
      <div className="page-header"><h1 className="page-title">Dashboard</h1></div>
      <div className="page-body">
        <div className="alert danger">
          <span className="alert-icon">⚠️</span>
          <div><strong>Backend connection error</strong><br />{error}<br /><small>Make sure the FastAPI server is running: <code>uvicorn app.main:app --reload</code></small></div>
        </div>
      </div>
    </div>
  );

  const total = stats?.total_assessments || 0;

  return (
    <div className="animate-fadeIn">
      <div className="page-header">
        <h1 className="page-title">Cardiac Risk Dashboard</h1>
        <p className="page-subtitle">Real-time monitoring overview — ML Research</p>
      </div>

      <div className="page-body">
        {/* Hero action */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(59,130,246,0.12) 0%, rgba(13,148,136,0.08) 100%)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-xl)',
          padding: '28px 32px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 24,
          flexWrap: 'wrap',
          gap: 16,
        }}>
          <div>
            <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: 20, fontWeight: 700, marginBottom: 6 }}>
              🫀 Start a New Assessment
            </h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>
              Enter patient ECG vitals and get instant ML-powered cardiac risk prediction.
            </p>
          </div>
          <button className="btn btn-primary btn-lg" id="btn-start-assessment" onClick={() => onNavigate('predict')}>
            ⚡ Run Prediction
          </button>
        </div>

        {/* Stat cards */}
        <div className="stat-grid">
          <div className="stat-card">
            <div className="stat-icon blue">📊</div>
            <div className="stat-info">
              <div className="value">{total}</div>
              <div className="label">Total Assessments</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon green">✅</div>
            <div className="stat-info">
              <div className="value" style={{ color: 'var(--risk-normal)' }}>{stats?.normal_count ?? 0}</div>
              <div className="label">Normal Risk</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon yellow">⚠️</div>
            <div className="stat-info">
              <div className="value" style={{ color: 'var(--risk-medium)' }}>{stats?.medium_count ?? 0}</div>
              <div className="label">Medium Risk</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon red">🚨</div>
            <div className="stat-info">
              <div className="value" style={{ color: 'var(--risk-critical)' }}>{stats?.critical_count ?? 0}</div>
              <div className="label">Critical Risk</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon teal">🎯</div>
            <div className="stat-info">
              <div className="value">{stats?.average_risk_score ?? 0}%</div>
              <div className="label">Avg Risk Score</div>
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 24 }}>
          {/* Distribution */}
          <div className="card">
            <div className="section-header">
              <div className="section-icon blue">📈</div>
              <h3 className="section-title">Risk Distribution</h3>
            </div>
            {total === 0 ? (
              <div className="empty-state" style={{ padding: '30px 0' }}>
                <span className="empty-icon">📭</span>
                <p>No assessments yet. Run the first prediction!</p>
              </div>
            ) : (
              <>
                <RiskBar label="Normal" count={stats.normal_count} total={total} color="var(--risk-normal)" />
                <RiskBar label="Medium Risk" count={stats.medium_count} total={total} color="var(--risk-medium)" />
                <RiskBar label="Critical" count={stats.critical_count} total={total} color="var(--risk-critical)" />
              </>
            )}
          </div>

          {/* Model info */}
          <div className="card">
            <div className="section-header">
              <div className="section-icon purple">🤖</div>
              <h3 className="section-title">ML Model Info</h3>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {[
                ['Algorithm', 'XGBoost (Gradient Boosting)'],
                ['Features', '18 ECG sliding-window features'],
                ['Input Intervals', 'RR · PP · QT (ms)'],
                ['Window Size', '5 readings (rolling stats)'],
                ['Score Range', '0–100 (probability × 100)'],
                ['Ensemble', 'Tabular + Waveform (CNNBiLSTM)'],
              ].map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, borderBottom: '1px solid var(--border)', paddingBottom: 8 }}>
                  <span style={{ color: 'var(--text-muted)' }}>{k}</span>
                  <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Recent assessments table */}
        <div className="card">
          <div className="section-header">
            <div className="section-icon teal">🕐</div>
            <h3 className="section-title">Recent Assessments</h3>
            <button className="btn btn-secondary btn-sm" style={{ marginLeft: 'auto' }} onClick={() => onNavigate('history')}>
              View All →
            </button>
          </div>

          {!stats?.recent_assessments?.length ? (
            <div className="empty-state">
              <span className="empty-icon">🔬</span>
              <h3>No assessments yet</h3>
              <p>Submit your first cardiac risk prediction to see results here.</p>
              <button className="btn btn-primary" style={{ marginTop: 8 }} onClick={() => onNavigate('predict')}>
                Start Now →
              </button>
            </div>
          ) : (
            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Patient</th>
                    <th>Demographics</th>
                    <th>Risk Level</th>
                    <th>Score</th>
                    <th>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.recent_assessments.map(rec => (
                    <RecentRow key={rec.id} rec={rec} onClick={() => onNavigate('history')} />
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
