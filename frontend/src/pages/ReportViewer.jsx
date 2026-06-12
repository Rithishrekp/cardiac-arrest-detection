import { useState, useEffect } from 'react';
import { api } from '../services/api';
import RiskCard from '../components/RiskCard';

export default function ReportViewer({ onNavigate }) {
  const [records, setRecords] = useState([]);
  const [selectedRecordId, setSelectedRecordId] = useState('');
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [selectedDetail, setSelectedDetail] = useState(null);
  const [error, setError] = useState(null);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    loadHistory();
  }, []);

  async function loadHistory() {
    setLoading(true);
    setError(null);
    try {
      const data = await api.history();
      setRecords(data);
      if (data.length > 0) {
        setSelectedRecordId(data[0].id);
        fetchReportDetail(data[0].id);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function fetchReportDetail(id) {
    if (!id) return;
    setDetailLoading(true);
    setError(null);
    try {
      const detail = await api.getRecord(id);
      setSelectedDetail(detail);
    } catch (err) {
      setError(err.message);
    } finally {
      setDetailLoading(false);
    }
  }

  function handleSelectChange(e) {
    const id = e.target.value;
    setSelectedRecordId(id);
    fetchReportDetail(id);
  }

  async function downloadReportPdf() {
    if (!selectedRecordId || !selectedDetail) return;
    setDownloading(true);
    try {
      const baseUrl = import.meta.env.VITE_API_URL || '/api';
      const res = await fetch(`${baseUrl}/generate-report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ record_id: Number(selectedRecordId) }),
      });
      if (!res.ok) throw new Error('Failed to generate report PDF from server');
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `cardiac_risk_report_${selectedDetail.record.patient_name.replace(/\s+/g, '_')}_${selectedRecordId}.pdf`;
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
        <h1 className="page-title">Medical Report Viewer</h1>
        <p className="page-subtitle">Select diagnostic records to view summaries and download PDF medical reports.</p>
      </div>

      <div className="page-body">
        {loading ? (
          <div className="loading-overlay">
            <div className="spinner" />
            <span>Loading records...</span>
          </div>
        ) : error ? (
          <div className="alert danger">
            <span>⚠️</span>
            <div>{error}</div>
          </div>
        ) : records.length === 0 ? (
          <div className="empty-state card">
            <span className="empty-icon">📭</span>
            <h3>No reports generated yet</h3>
            <p>Go to the Predict page to submit patient vitals and generate your first screening report.</p>
            <button className="btn btn-primary" onClick={() => onNavigate('predict')}>
              Run Prediction →
            </button>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Dropdown Selector Card */}
            <div className="card" style={{ display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap' }}>
              <label htmlFor="report-selector" style={{ fontWeight: '600', fontSize: '14px', color: 'var(--text-secondary)' }}>
                Select Patient Diagnostic Record:
              </label>
              <select
                id="report-selector"
                value={selectedRecordId}
                onChange={handleSelectChange}
                style={{
                  padding: '8px 12px',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border)',
                  background: 'var(--bg-glass-input)',
                  color: 'var(--text-primary)',
                  fontWeight: '600',
                  minWidth: '280px'
                }}
              >
                {records.map(rec => (
                  <option key={rec.id} value={rec.id}>
                    #{rec.id} - {rec.patient_name} ({rec.patient_id})
                  </option>
                ))}
              </select>

              <button
                className="btn btn-primary"
                onClick={downloadReportPdf}
                disabled={detailLoading || downloading}
                style={{ marginLeft: 'auto' }}
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

            {/* Display Report Summary */}
            {detailLoading ? (
              <div className="loading-overlay" style={{ padding: '60px 0' }}>
                <div className="spinner" />
                <span>Loading report details...</span>
              </div>
            ) : selectedDetail ? (
              <div style={{ animation: 'fadeIn 0.5s ease' }}>
                <div style={{
                  background: 'rgba(30,58,138,0.12)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-lg)',
                  padding: '16px 20px',
                  marginBottom: '20px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <div>
                    <h3 style={{ margin: 0, fontWeight: 700, fontSize: '15px' }}>
                      Diagnostic Assessment Report Summary (Record #{selectedDetail.record.id})
                    </h3>
                    <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: 'var(--text-secondary)' }}>
                      Evaluated on: {new Date(selectedDetail.record.created_at).toLocaleString()}
                    </p>
                  </div>
                  
                  {/* Model Confidence Tag */}
                  <div style={{
                    background: 'rgba(59,130,246,0.15)',
                    border: '1px solid rgba(59,130,246,0.3)',
                    color: '#3B82F6',
                    padding: '6px 12px',
                    borderRadius: '20px',
                    fontSize: '12px',
                    fontWeight: 700
                  }}>
                    Model Confidence: {selectedDetail.record.model_confidence || 95.0}%
                  </div>
                </div>

                <RiskCard
                  record={selectedDetail.record}
                  suggestions={selectedDetail.suggestions}
                  contributions={selectedDetail.contributions}
                  disclaimer={selectedDetail.disclaimer}
                />
              </div>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}
