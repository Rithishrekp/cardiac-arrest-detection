const RISK_CONFIG = {
  normal: {
    label: 'Normal / Low Risk',
    emoji: '✅',
    className: 'normal',
    description: 'ECG intervals and vitals appear within normal range. Continue routine athletic monitoring.',
  },
  medium: {
    label: 'Moderate Risk',
    emoji: '⚠️',
    className: 'medium',
    description: 'Elevated cardiac risk detected. A diagnostic clinical consultation is recommended.',
  },
  critical: {
    label: 'High Risk / Alert',
    emoji: '🚨',
    className: 'critical',
    description: 'Critical cardiac abnormalities detected. Seek immediate clinical cardiology evaluation.',
  },
};

export default function RiskCard({ record, suggestions = [], contributions = [], disclaimer }) {
  const config = RISK_CONFIG[record.risk_level] || RISK_CONFIG.normal;
  const score = Math.round(record.risk_score * 10) / 10;

  return (
    <div className="result-container" style={{ display: 'grid', gap: '1.5rem' }}>
      
      {/* 1. Risk Meter Gauge Card */}
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
          <div className="risk-score-label">Predicted Cardiac Arrest Probability</div>
        </div>

        <div className="risk-progress-bar">
          <div
            className="risk-progress-fill"
            style={{ width: `${Math.min(score, 100)}%` }}
          />
        </div>

        <p className="risk-description">{config.description}</p>
      </div>

      {/* 2. Feature Contribution Dashboard (Phase 4 Explainability) */}
      {contributions.length > 0 && (
        <div className="card feature-explanation-card" style={{ padding: '1.5rem' }}>
          <h3 style={{ marginTop: 0, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            🔬 Patient Feature Contribution Dashboard
          </h3>
          <p className="form-hint" style={{ marginBottom: '1.25rem' }}>
            Top physiological and temporal factors that drove this patient's risk classification:
          </p>
          <div className="contributions-list" style={{ display: 'grid', gap: '1rem' }}>
            {contributions.map((c, i) => (
              <div key={i} className="contribution-item">
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
                  <span style={{ fontWeight: '600' }}>
                    {c.feature} <span style={{ fontWeight: 'normal', color: 'var(--text-light)', fontSize: '0.8rem' }}>({c.value})</span>
                  </span>
                  <span style={{ fontWeight: 'bold', color: c.direction === 'increase' ? 'var(--critical)' : 'var(--primary)' }}>
                    {c.contribution}% ({c.direction === 'increase' ? 'High' : 'Low'})
                  </span>
                </div>
                <div className="progress-bar-bg" style={{ height: '8px', background: 'var(--border)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div 
                    className="progress-bar-fill" 
                    style={{ 
                      height: '100%', 
                      width: `${c.contribution}%`, 
                      background: c.direction === 'increase' ? 'linear-gradient(90deg, #f59e0b, #ef4444)' : 'linear-gradient(90deg, #10b981, #3b82f6)',
                      borderRadius: '4px' 
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 3. Patient Vitals Details Grid */}
      <div className="card patient-summary">
        <h3>Clinical Metrics Summary</h3>
        <div className="detail-grid">
          <Detail label="Name" value={record.patient_name} />
          <Detail label="Patient ID" value={record.patient_id} />
          <Detail label="Age" value={`${record.age} years`} />
          <Detail label="Gender" value={record.gender} />
          <Detail label="Sport" value={record.sport_type} />
          
          <Detail label="Weight" value={`${record.weight} kg`} />
          <Detail label="Height" value={`${record.height} cm`} />
          <Detail label="BMI" value={`${record.bmi}`} />
          <Detail label="Heart Rate" value={`${record.heart_rate} bpm`} />
          <Detail label="Systolic BP" value={`${record.systolic_bp} mmHg`} />
          <Detail label="Diastolic BP" value={`${record.diastolic_bp} mmHg`} />
          <Detail label="Mean Art. Press." value={`${record.mean_arterial_pressure} mmHg`} />

          <Detail label="RR Interval" value={`${record.rr_interval} ms`} />
          <Detail label="PP Interval" value={`${record.pp_interval} ms`} />
          <Detail label="QT Interval" value={`${record.qt_interval} ms`} />
          <Detail label="QTc Interval" value={`${record.qtc_interval} ms`} />
          <Detail label="QRS Duration" value={`${record.qrs_duration} ms`} />
          <Detail label="PQ Interval" value={`${record.pq_interval} ms`} />
          
          <Detail label="Family Heart History" value={record.family_history_heart_disease ? "Yes 🔴" : "No"} />
          <Detail label="Personal Heart History" value={record.personal_history_heart_disease ? "Yes 🔴" : "No"} />
          <Detail label="Syncope / Fainting" value={record.syncope ? "Yes 🔴" : "No"} />
          <Detail label="Pectus Excavatum" value={record.pectus_excavatum ? "Yes 🔴" : "No"} />
          
          <Detail
            label="Assessed At"
            value={new Date(record.created_at).toLocaleString()}
          />
        </div>
      </div>

      {/* 4. Actionable Suggestions */}
      {suggestions.length > 0 && (
        <div className="card suggestions-card">
          <h3>Health & Diagnostic Recommendations</h3>
          <ul className="suggestions-list">
            {suggestions.map((tip, i) => (
              <li key={i}>{tip}</li>
            ))}
          </ul>
        </div>
      )}

      {/* 5. Medical Disclaimer */}
      <div className="disclaimer-banner">
        <span>⚕️</span>
        <p>{disclaimer || 'Important Note: This is only an ML-based prediction and not a medical diagnosis.'}</p>
      </div>
    </div>
  );
}

function Detail({ label, value }) {
  return (
    <div className="detail-item">
      <span className="detail-label">{label}</span>
      <span className="detail-value" style={{ fontWeight: '600' }}>{value}</span>
    </div>
  );
}
