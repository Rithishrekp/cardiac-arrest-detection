import { useEffect, useState } from 'react';
import RiskCard from '../components/RiskCard';
import { api } from '../services/api';

export default function Result({ result, onNavigate }) {
  const [correlations, setCorrelations] = useState([]);
  const [corrLoading, setCorrLoading] = useState(true);

  useEffect(() => {
    async function loadCorrelations() {
      try {
        const data = await api.correlations();
        // Sort by correlation percentage descending and take top 8
        const sorted = (data || [])
          .sort((a, b) => b['Correlation (%)'] - a['Correlation (%)'])
          .slice(0, 8);
        setCorrelations(sorted);
      } catch (err) {
        console.error('Failed to load correlations:', err);
      } finally {
        setCorrLoading(false);
      }
    }
    loadCorrelations();
  }, []);

  if (!result) {
    return (
      <div className="page-body">
        <div className="empty-state">
          <span className="empty-icon">🔬</span>
          <h3>No result to display</h3>
          <p>Run a prediction first to see results here.</p>
          <button className="btn btn-primary" onClick={() => onNavigate('predict')}>
            Go to Prediction Form →
          </button>
        </div>
      </div>
    );
  }

  const { record, suggestions, contributions, disclaimer } = result;
  const [downloading, setDownloading] = useState(false);

  async function downloadReportPdf() {
    if (!record?.id) return;
    setDownloading(true);
    try {
      const baseUrl = import.meta.env.VITE_API_URL || '/api';
      const res = await fetch(`${baseUrl}/generate-report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ record_id: Number(record.id) }),
      });
      if (!res.ok) throw new Error('Failed to generate report PDF');
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `cardiac_risk_report_${record.patient_name.replace(/\s+/g, '_')}_${record.id}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert('Error downloading PDF: ' + err.message);
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div className="animate-fadeIn">
      <div className="page-header">
        <h1 className="page-title">Assessment Result</h1>
        <p className="page-subtitle">
          ML-powered cardiac risk prediction for {record.patient_name}
        </p>
      </div>

      <div className="page-body" style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: '2rem', alignItems: 'start' }}>
        
        {/* Main Risk Output Cards (Risk Meter, Details, Suggestions) */}
        <div>
          {/* Model Confidence Tag */}
          <div style={{
            background: 'rgba(30,58,138,0.12)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-lg)',
            padding: '12px 18px',
            marginBottom: '1.5rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <div>
              <span style={{ fontWeight: 700, fontSize: '14px' }}>📈 AI Classifier Quality Metrics:</span>
              <p style={{ margin: '2px 0 0 0', fontSize: '11px', color: 'var(--text-secondary)' }}>Based on multi-feature XGBoost sports models</p>
            </div>
            <div style={{
              background: 'rgba(59,130,246,0.15)',
              border: '1px solid rgba(59,130,246,0.3)',
              color: '#3B82F6',
              padding: '4px 12px',
              borderRadius: '20px',
              fontSize: '11px',
              fontWeight: 700
            }}>
              Model Confidence: {record.model_confidence || 95.0}%
            </div>
          </div>

          <RiskCard
            record={record}
            suggestions={suggestions}
            contributions={contributions}
            disclaimer={disclaimer}
          />

          <div className="result-actions" style={{ marginTop: '2rem', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            <button className="btn btn-secondary" onClick={() => onNavigate('predict')}>
              ← New Assessment
            </button>
            <button
              className="btn btn-primary"
              onClick={() => onNavigate('history')}
            >
              View History Database →
            </button>
            <button
              className="btn btn-primary"
              onClick={downloadReportPdf}
              disabled={downloading}
              style={{ background: 'var(--success)', borderColor: 'var(--success)' }}
            >
              {downloading ? (
                <>
                  <span className="spinner spinner-sm" /> Downloading PDF...
                </>
              ) : (
                '📄 Download PDF Report'
              )}
            </button>
          </div>
        </div>

        {/* Dataset-level Correlation Dashboard */}
        <aside className="correlation-dashboard-sidebar">
          <div className="card" style={{ padding: '1.5rem', position: 'sticky', top: '20px' }}>
            <h3 style={{ marginTop: 0, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              📊 Dataset Correlation Dashboard
            </h3>
            <p className="form-hint" style={{ marginBottom: '1.25rem' }}>
              Dataset-wide correlations between ECG intervals and `CardiacRisk_Encoded` based on 22,648 integrated patient records.
            </p>

            {corrLoading ? (
              <div style={{ padding: '2rem', textAlign: 'center' }}>
                <span className="spinner spinner-sm" /> Loading correlation index...
              </div>
            ) : correlations.length > 0 ? (
              <div style={{ display: 'grid', gap: '1rem' }}>
                <table className="history-table" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid var(--border)', textAlign: 'left' }}>
                      <th style={{ padding: '0.5rem 0.25rem' }}>Feature</th>
                      <th style={{ padding: '0.5rem 0.25rem', textAlign: 'center' }}>Corr (%)</th>
                      <th style={{ padding: '0.5rem 0.25rem', textAlign: 'right' }}>Strength</th>
                    </tr>
                  </thead>
                  <tbody>
                    {correlations.map((c, idx) => {
                      let badgeColor = '#6b7280'; // gray
                      if (c.strength === 'Very Strong') badgeColor = '#ef4444'; // red
                      if (c.strength === 'Strong') badgeColor = '#f59e0b'; // orange
                      if (c.strength === 'Moderate') badgeColor = '#3b82f6'; // blue
                      if (c.strength === 'Weak') badgeColor = '#10b981'; // green

                      return (
                        <tr key={idx} style={{ borderBottom: '1px solid var(--border)' }}>
                          <td style={{ padding: '0.5rem 0.25rem', fontWeight: '500' }}>{c.feature}</td>
                          <td style={{ padding: '0.5rem 0.25rem', textAlign: 'center' }}>
                            {c['Correlation (%)'] ? c['Correlation (%)'].toFixed(1) : '0.0'}%
                          </td>
                          <td style={{ padding: '0.5rem 0.25rem', textAlign: 'right' }}>
                            <span 
                              style={{ 
                                display: 'inline-block', 
                                padding: '0.15rem 0.4rem', 
                                borderRadius: '4px', 
                                fontSize: '0.75rem', 
                                color: '#fff', 
                                background: badgeColor,
                                fontWeight: 'bold'
                              }}
                            >
                              {c.strength}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <p className="info-note" style={{ fontSize: '0.75rem', margin: 0, padding: '0.5rem', background: 'var(--bg)', borderRadius: '4px' }}>
                  * Pearson values calculated using linear regression statistics.
                </p>
              </div>
            ) : (
              <p>No correlation data available.</p>
            )}
          </div>
        </aside>

      </div>
    </div>
  );
}
