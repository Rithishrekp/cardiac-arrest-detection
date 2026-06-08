const RISK_CONFIG = {
  normal: {
    label: 'Low Risk',
    emoji: '✅',
    className: 'normal',
    description: 'ECG intervals appear within normal range. Continue routine monitoring.',
  },
  medium: {
    label: 'Medium Risk',
    emoji: '⚠️',
    className: 'medium',
    description: 'Elevated cardiac risk detected. Clinical follow-up is recommended.',
  },
  critical: {
    label: 'High Risk',
    emoji: '🚨',
    className: 'critical',
    description: 'Critical cardiac risk detected. Seek immediate medical attention.',
  },
};

export default function RiskCard({ record, suggestions = [], disclaimer }) {
  const config = RISK_CONFIG[record.risk_level] || RISK_CONFIG.normal;
  const score = Math.round(record.risk_score * 10) / 10;

  return (
    <div className="result-container">
      <div className={`risk-card ${config.className}`}>
        <div className="risk-card-header">
          <span className="risk-emoji">{config.emoji}</span>
          <div>
            <div className="risk-category">{config.label}</div>
            <div className="risk-model-label">{record.risk_label}</div>
          </div>
        </div>

        <div className="risk-score-display">
          <div className="risk-score-value">{score}%</div>
          <div className="risk-score-label">Risk Probability</div>
        </div>

        <div className="risk-progress-bar">
          <div
            className="risk-progress-fill"
            style={{ width: `${Math.min(score, 100)}%` }}
          />
        </div>

        <p className="risk-description">{config.description}</p>
      </div>

      <div className="card patient-summary">
        <h3>Patient Details</h3>
        <div className="detail-grid">
          <Detail label="Name" value={record.patient_name} />
          <Detail label="Patient ID" value={record.patient_id} />
          <Detail label="Age" value={`${record.age} years`} />
          <Detail label="Gender" value={record.gender} />
          <Detail label="RR Interval" value={`${record.rr_interval} ms`} />
          <Detail label="PP Interval" value={`${record.pp_interval} ms`} />
          <Detail label="QT Interval" value={`${record.qt_interval} ms`} />
          <Detail
            label="Assessed At"
            value={new Date(record.created_at).toLocaleString()}
          />
        </div>
      </div>

      {suggestions.length > 0 && (
        <div className="card suggestions-card">
          <h3>Health Recommendations</h3>
          <ul className="suggestions-list">
            {suggestions.map((tip, i) => (
              <li key={i}>{tip}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="disclaimer-banner">
        <span>⚕️</span>
        <p>{disclaimer || 'This is only an ML-based prediction and not a medical diagnosis.'}</p>
      </div>
    </div>
  );
}

function Detail({ label, value }) {
  return (
    <div className="detail-item">
      <span className="detail-label">{label}</span>
      <span className="detail-value">{value}</span>
    </div>
  );
}
