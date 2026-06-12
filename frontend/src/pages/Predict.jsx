import { useState } from 'react';
import { api } from '../services/api';

const INITIAL_FORM = {
  patient_id: '',
  patient_name: '',
  age: '',
  gender: 'Male',
  weight: '',
  height: '',
  heart_rate: '',
  systolic_bp: '',
  diastolic_bp: '',
  sport_type: 'ATH',
  rr_interval: '',
  pp_interval: '',
  qt_interval: '',
  qtc_interval: '',
  qrs_duration: '',
  pq_interval: '',
  family_history_heart_disease: false,
  personal_history_heart_disease: false,
  syncope: false,
  pectus_excavatum: false,

  // Upgraded dynamic features
  person_type: 'sports',
  race: 'White',
  years_sports_experience: '',
  training_hours_per_week: '',
  competition_level: 'Amateur',
  recent_intense_exercise: false,
  previous_collapse_during_sports: false,
  fatigue_during_exercise: false,
  loss_of_consciousness_during_exercise: false,
  shortness_of_breath_during_exercise: false,
  chest_pain_during_exercise: false,
  dizziness_during_exercise: false,
  palpitations_during_exercise: false,
  
  workout_frequency: '',
  heavy_weight_training: false,
  cardio_frequency: '',
  steroid_usage: false,
  supplement_usage: false,
  pre_workout_usage: false,
  energy_drink_consumption: false,
  dehydration_episodes: false,
  overtraining_symptoms: false,
  recent_fainting_episodes: false,
  
  family_history_cardiac_arrest: false,
  family_history_sudden_death: false,
  hypertension: false,
  diabetes: false,
  known_heart_disease: false,
  congenital_heart_disease: false,
  smoking: false,
  alcohol_consumption: false,
  previous_cardiac_problems: false,
  current_medication: '',
};

const SPORT_TYPES = [
  { value: 'ATH', label: 'Athletics / Running' },
  { value: 'AMF', label: 'American Football' },
  { value: 'VOL', label: 'Volleyball' },
  { value: 'CYC', label: 'Cycling' },
  { value: 'RUN', label: 'Running / Sprinting' },
  { value: 'TRE', label: 'Trekking / Climbing' },
  { value: 'FUT', label: 'Football / Soccer' },
  { value: 'CFIT', label: 'CrossFit / Powerlifting' },
];

const RACES = ['White', 'Asian', 'Black', 'Hispanic', 'Other'];

function validate(form) {
  const errors = {};

  if (!form.patient_id.trim()) errors.patient_id = 'Patient ID is required';
  if (!form.patient_name.trim()) errors.patient_name = 'Patient name is required';

  const age = Number(form.age);
  if (!form.age || isNaN(age) || age < 1 || age > 120) {
    errors.age = 'Age must be between 1 and 120';
  }

  const weight = Number(form.weight);
  if (!form.weight || isNaN(weight) || weight < 5 || weight > 300) {
    errors.weight = 'Weight must be between 5 and 300 kg';
  }

  const height = Number(form.height);
  if (!form.height || isNaN(height) || height < 50 || height > 250) {
    errors.height = 'Height must be between 50 and 250 cm';
  }

  const hr = Number(form.heart_rate);
  if (!form.heart_rate || isNaN(hr) || hr < 30 || hr > 220) {
    errors.heart_rate = 'Heart rate must be 30–220 bpm';
  }

  const sbp = Number(form.systolic_bp);
  if (!form.systolic_bp || isNaN(sbp) || sbp < 50 || sbp > 250) {
    errors.systolic_bp = 'Systolic BP must be 50–250 mmHg';
  }

  const dbp = Number(form.diastolic_bp);
  if (!form.diastolic_bp || isNaN(dbp) || dbp < 35 || dbp > 160) {
    errors.diastolic_bp = 'Diastolic BP must be 35–160 mmHg';
  }

  const rr = Number(form.rr_interval);
  if (!form.rr_interval || isNaN(rr) || rr < 100 || rr > 2000) {
    errors.rr_interval = 'RR interval must be 100–2000 ms';
  }

  const pp = Number(form.pp_interval);
  if (!form.pp_interval || isNaN(pp) || pp < 100 || pp > 2000) {
    errors.pp_interval = 'PP interval must be 100–2000 ms';
  }

  const qt = Number(form.qt_interval);
  if (!form.qt_interval || isNaN(qt) || qt < 100 || qt > 800) {
    errors.qt_interval = 'QT interval must be 100–800 ms';
  }

  const qtc = Number(form.qtc_interval);
  if (!form.qtc_interval || isNaN(qtc) || qtc < 100 || qtc > 800) {
    errors.qtc_interval = 'QTc interval must be 100–800 ms';
  }

  const qrs = Number(form.qrs_duration);
  if (!form.qrs_duration || isNaN(qrs) || qrs < 20 || qrs > 250) {
    errors.qrs_duration = 'QRS duration must be 20–250 ms';
  }

  const pq = Number(form.pq_interval);
  if (!form.pq_interval || isNaN(pq) || pq < 20 || pq > 400) {
    errors.pq_interval = 'PQ interval must be 20–400 ms';
  }

  return errors;
}

export default function Predict({ onResult }) {
  const [form, setForm] = useState(INITIAL_FORM);
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState(null);
  const [activeTab, setActiveTab] = useState('demographics');

  function handleChange(e) {
    const { name, value, type, checked } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
    setErrors((prev) => ({ ...prev, [name]: undefined }));
    setApiError(null);
  }

  function fillSample(type) {
    const isGym = type === 'gym';
    setForm({
      patient_id: isGym ? 'G-8809' : 'P-2024-ATH',
      patient_name: isGym ? 'Amit Patel' : 'Vikram Singh',
      age: isGym ? '31' : '26',
      gender: 'Male',
      weight: isGym ? '82' : '74',
      height: isGym ? '178' : '180',
      heart_rate: isGym ? '76' : '68',
      systolic_bp: isGym ? '135' : '124',
      diastolic_bp: isGym ? '88' : '82',
      sport_type: isGym ? 'CFIT' : 'ATH',
      rr_interval: '880',
      pp_interval: '875',
      qt_interval: '390',
      qtc_interval: '415',
      qrs_duration: '98',
      pq_interval: '155',
      family_history_heart_disease: !isGym,
      personal_history_heart_disease: false,
      syncope: isGym,
      pectus_excavatum: false,

      // dynamic choices
      person_type: isGym ? 'gym' : 'sports',
      race: 'Asian',
      years_sports_experience: isGym ? '0' : '6',
      training_hours_per_week: isGym ? '0' : '10',
      competition_level: 'Professional',
      recent_intense_exercise: true,
      previous_collapse_during_sports: false,
      fatigue_during_exercise: isGym,
      loss_of_consciousness_during_exercise: false,
      shortness_of_breath_during_exercise: isGym,
      chest_pain_during_exercise: false,
      dizziness_during_exercise: isGym,
      palpitations_during_exercise: false,
      
      workout_frequency: isGym ? '4' : '0',
      heavy_weight_training: isGym,
      cardio_frequency: isGym ? '2' : '0',
      steroid_usage: isGym, // steroids active for gym sample
      supplement_usage: true,
      pre_workout_usage: isGym,
      energy_drink_consumption: isGym,
      dehydration_episodes: false,
      overtraining_symptoms: isGym,
      recent_fainting_episodes: isGym,
      
      family_history_cardiac_arrest: false,
      family_history_sudden_death: false,
      hypertension: isGym,
      diabetes: false,
      known_heart_disease: false,
      congenital_heart_disease: false,
      smoking: isGym,
      alcohol_consumption: true,
      previous_cardiac_problems: false,
      current_medication: isGym ? 'Crestor 10mg' : 'None',
    });
    setErrors({});
    setApiError(null);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    const validationErrors = validate(form);
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      if (validationErrors.patient_id || validationErrors.patient_name || validationErrors.age || validationErrors.weight || validationErrors.height) {
        setActiveTab('demographics');
      } else {
        setActiveTab('vitals');
      }
      return;
    }

    setLoading(true);
    setApiError(null);

    try {
      const payload = {
        patient_id: form.patient_id.trim(),
        patient_name: form.patient_name.trim(),
        age: Number(form.age),
        gender: form.gender,
        weight: Number(form.weight),
        height: Number(form.height),
        heart_rate: Number(form.heart_rate),
        systolic_bp: Number(form.systolic_bp),
        diastolic_bp: Number(form.diastolic_bp),
        sport_type: form.sport_type,
        rr_interval: Number(form.rr_interval),
        pp_interval: Number(form.pp_interval),
        qt_interval: Number(form.qt_interval),
        qtc_interval: Number(form.qtc_interval),
        qrs_duration: Number(form.qrs_duration),
        pq_interval: Number(form.pq_interval),
        family_history_heart_disease: form.family_history_heart_disease ? 1.0 : 0.0,
        personal_history_heart_disease: form.personal_history_heart_disease ? 1.0 : 0.0,
        syncope: form.syncope ? 1.0 : 0.0,
        pectus_excavatum: form.pectus_excavatum ? 1.0 : 0.0,
        
        person_type: form.person_type,
        years_sports_experience: form.person_type === 'sports' ? Number(form.years_sports_experience) || 0.0 : 0.0,
        training_hours_per_week: form.person_type === 'sports' ? Number(form.training_hours_per_week) || 0.0 : 0.0,
        competition_level: form.person_type === 'sports' ? form.competition_level : 'Amateur',
        recent_intense_exercise: form.recent_intense_exercise ? 1.0 : 0.0,
        previous_collapse_during_sports: form.previous_collapse_during_sports ? 1.0 : 0.0,
        fatigue_during_exercise: form.fatigue_during_exercise ? 1.0 : 0.0,
        loss_of_consciousness_during_exercise: form.loss_of_consciousness_during_exercise ? 1.0 : 0.0,
        shortness_of_breath_during_exercise: form.shortness_of_breath_during_exercise ? 1.0 : 0.0,
        chest_pain_during_exercise: form.chest_pain_during_exercise ? 1.0 : 0.0,
        dizziness_during_exercise: form.dizziness_during_exercise ? 1.0 : 0.0,
        palpitations_during_exercise: form.palpitations_during_exercise ? 1.0 : 0.0,
        
        workout_frequency: form.person_type === 'gym' ? Number(form.workout_frequency) || 0.0 : 0.0,
        heavy_weight_training: form.heavy_weight_training ? 1.0 : 0.0,
        cardio_frequency: form.person_type === 'gym' ? Number(form.cardio_frequency) || 0.0 : 0.0,
        steroid_usage: form.steroid_usage ? 1.0 : 0.0,
        supplement_usage: form.supplement_usage ? 1.0 : 0.0,
        pre_workout_usage: form.pre_workout_usage ? 1.0 : 0.0,
        energy_drink_consumption: form.energy_drink_consumption ? 1.0 : 0.0,
        dehydration_episodes: form.dehydration_episodes ? 1.0 : 0.0,
        overtraining_symptoms: form.overtraining_symptoms ? 1.0 : 0.0,
        recent_fainting_episodes: form.recent_fainting_episodes ? 1.0 : 0.0,
        
        family_history_cardiac_arrest: form.family_history_cardiac_arrest ? 1.0 : 0.0,
        family_history_sudden_death: form.family_history_sudden_death ? 1.0 : 0.0,
        hypertension: form.hypertension ? 1.0 : 0.0,
        diabetes: form.diabetes ? 1.0 : 0.0,
        known_heart_disease: form.known_heart_disease ? 1.0 : 0.0,
        congenital_heart_disease: form.congenital_heart_disease ? 1.0 : 0.0,
        smoking: form.smoking ? 1.0 : 0.0,
        alcohol_consumption: form.alcohol_consumption ? 1.0 : 0.0,
        previous_cardiac_problems: form.previous_cardiac_problems ? 1.0 : 0.0,
        current_medication: form.current_medication.trim(),
      };

      const result = await api.predict(payload);
      onResult(result);
    } catch (err) {
      setApiError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="animate-fadeIn">
      <div className="page-header">
        <h1 className="page-title">Cardiac Assessment Form</h1>
        <p className="page-subtitle">
          Submit full demographic, physiological, lifestyle, and ECG signals to evaluate sudden cardiac arrest risk.
        </p>
      </div>

      <div className="page-body">
        <div className="form-layout">
          <form className="card prediction-form" onSubmit={handleSubmit} noValidate>
            
            {/* Person Type Switcher */}
            <div style={{
              background: 'rgba(255,255,255,0.05)',
              padding: '12px 18px',
              borderRadius: 'var(--radius-lg)',
              border: '1px solid var(--border)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '1.5rem'
            }}>
              <span style={{ fontWeight: 700, fontSize: '14px' }}>📋 Choose Subject Profile:</span>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  type="button"
                  onClick={() => setForm(f => ({ ...f, person_type: 'sports' }))}
                  className={`btn btn-sm ${form.person_type === 'sports' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ borderRadius: '15px', padding: '5px 16px' }}
                >
                  🏃 Sports Person
                </button>
                <button
                  type="button"
                  onClick={() => setForm(f => ({ ...f, person_type: 'gym' }))}
                  className={`btn btn-sm ${form.person_type === 'gym' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ borderRadius: '15px', padding: '5px 16px' }}
                >
                  💪 Gym Individual
                </button>
              </div>
            </div>

            {/* Form Steps / Tab Navigation */}
            <div className="form-tabs" style={{ display: 'flex', borderBottom: '1px solid var(--border)', marginBottom: '1.5rem' }}>
              <button
                type="button"
                className={`tab-btn ${activeTab === 'demographics' ? 'active' : ''}`}
                onClick={() => setActiveTab('demographics')}
                style={{ padding: '0.75rem 1.25rem', border: 'none', background: 'none', borderBottom: activeTab === 'demographics' ? '3px solid var(--primary)' : 'none', cursor: 'pointer', fontWeight: '600' }}
              >
                👤 Profile & Demographics
              </button>
              <button
                type="button"
                className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`}
                onClick={() => setActiveTab('history')}
                style={{ padding: '0.75rem 1.25rem', border: 'none', background: 'none', borderBottom: activeTab === 'history' ? '3px solid var(--primary)' : 'none', cursor: 'pointer', fontWeight: '600' }}
              >
                📋 Medical & Family History
              </button>
              <button
                type="button"
                className={`tab-btn ${activeTab === 'vitals' ? 'active' : ''}`}
                onClick={() => setActiveTab('vitals')}
                style={{ padding: '0.75rem 1.25rem', border: 'none', background: 'none', borderBottom: activeTab === 'vitals' ? '3px solid var(--primary)' : 'none', cursor: 'pointer', fontWeight: '600' }}
              >
                ⚡ Vitals, ECG & Symptoms
              </button>
            </div>

            {/* TAB 1: Demographics */}
            {activeTab === 'demographics' && (
              <div className="form-section animate-fadeIn">
                <h3 className="form-section-title">Patient Profile</h3>
                <div className="form-grid">
                  <FormField
                    label="Patient ID / MRN"
                    name="patient_id"
                    value={form.patient_id}
                    onChange={handleChange}
                    error={errors.patient_id}
                    placeholder="e.g. P-1001"
                    required
                  />
                  <FormField
                    label="Full Name"
                    name="patient_name"
                    value={form.patient_name}
                    onChange={handleChange}
                    error={errors.patient_name}
                    placeholder="e.g. Vikram Singh"
                    required
                  />
                  <FormField
                    label="Age (years)"
                    name="age"
                    type="number"
                    value={form.age}
                    onChange={handleChange}
                    error={errors.age}
                    placeholder="1–120"
                    required
                  />
                  <div className="form-field">
                    <label htmlFor="gender">Gender</label>
                    <select id="gender" name="gender" value={form.gender} onChange={handleChange}>
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>
                  <FormField
                    label="Weight (kg)"
                    name="weight"
                    type="number"
                    value={form.weight}
                    onChange={handleChange}
                    error={errors.weight}
                    placeholder="5–300"
                    required
                  />
                  <FormField
                    label="Height (cm)"
                    name="height"
                    type="number"
                    value={form.height}
                    onChange={handleChange}
                    error={errors.height}
                    placeholder="50–250"
                    required
                  />
                  <div className="form-field">
                    <label htmlFor="race">Race / Ethnicity</label>
                    <select id="race" name="race" value={form.race} onChange={handleChange}>
                      {RACES.map(r => <option key={r} value={r}>{r}</option>)}
                    </select>
                  </div>
                  <div className="form-field">
                    <label htmlFor="sport_type">Primary Sport</label>
                    <select id="sport_type" name="sport_type" value={form.sport_type} onChange={handleChange}>
                      {SPORT_TYPES.map((s) => (
                        <option key={s.value} value={s.value}>
                          {s.label}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* DYNAMIC SUB-SECTION */}
                {form.person_type === 'sports' ? (
                  <div style={{ marginTop: '1.5rem', borderTop: '1px solid var(--border)', paddingTop: '1.5rem' }}>
                    <h3 className="form-section-title">🏋️ Sports Person Profiling</h3>
                    <div className="form-grid">
                      <FormField
                        label="Years of Sports Experience"
                        name="years_sports_experience"
                        type="number"
                        value={form.years_sports_experience}
                        onChange={handleChange}
                        placeholder="e.g. 5"
                      />
                      <FormField
                        label="Training Hours Per Week"
                        name="training_hours_per_week"
                        type="number"
                        value={form.training_hours_per_week}
                        onChange={handleChange}
                        placeholder="e.g. 8"
                      />
                      <div className="form-field">
                        <label htmlFor="competition_level">Competition Level</label>
                        <select id="competition_level" name="competition_level" value={form.competition_level} onChange={handleChange}>
                          <option value="Amateur">Amateur / Recreational</option>
                          <option value="Professional">Professional / Elite</option>
                        </select>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div style={{ marginTop: '1.5rem', borderTop: '1px solid var(--border)', paddingTop: '1.5rem' }}>
                    <h3 className="form-section-title">💪 Gym Individual Profiling</h3>
                    <div className="form-grid">
                      <FormField
                        label="Workout Frequency (days/week)"
                        name="workout_frequency"
                        type="number"
                        value={form.workout_frequency}
                        onChange={handleChange}
                        placeholder="e.g. 4"
                      />
                      <FormField
                        label="Cardio Frequency (days/week)"
                        name="cardio_frequency"
                        type="number"
                        value={form.cardio_frequency}
                        onChange={handleChange}
                        placeholder="e.g. 2"
                      />
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* TAB 2: Medical History */}
            {activeTab === 'history' && (
              <div className="form-section animate-fadeIn">
                <h3 className="form-section-title">Genetics & Family History</h3>
                <div className="checkbox-list" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '0.5rem', marginBottom: '1.5rem' }}>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="family_history_heart_disease" checked={form.family_history_heart_disease} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Family History of Heart Disease</strong></div>
                  </label>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="family_history_cardiac_arrest" checked={form.family_history_cardiac_arrest} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Family History of Cardiac Arrest</strong></div>
                  </label>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="family_history_sudden_death" checked={form.family_history_sudden_death} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Family History of Sudden Death</strong></div>
                  </label>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="personal_history_heart_disease" checked={form.personal_history_heart_disease} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Personal History of Heart Disease</strong></div>
                  </label>
                </div>

                <h3 className="form-section-title" style={{ borderTop: '1px solid var(--border)', paddingTop: '1.5rem' }}>Anamnesis & Medical Vitals</h3>
                <div className="checkbox-list" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '0.5rem', marginBottom: '1.5rem' }}>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="hypertension" checked={form.hypertension} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Hypertension</strong></div>
                  </label>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="diabetes" checked={form.diabetes} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Diabetes</strong></div>
                  </label>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="known_heart_disease" checked={form.known_heart_disease} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Known Heart Disease</strong></div>
                  </label>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="congenital_heart_disease" checked={form.congenital_heart_disease} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Congenital Heart Disease</strong></div>
                  </label>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="smoking" checked={form.smoking} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Active Smoking</strong></div>
                  </label>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="alcohol_consumption" checked={form.alcohol_consumption} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Alcohol Consumption</strong></div>
                  </label>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="previous_cardiac_problems" checked={form.previous_cardiac_problems} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Previous Cardiac Problems</strong></div>
                  </label>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="pectus_excavatum" checked={form.pectus_excavatum} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Pectus Excavatum (Sunken Chest)</strong></div>
                  </label>
                </div>

                <div className="form-field" style={{ marginTop: '1rem' }}>
                  <label htmlFor="current_medication">Current Medication / Dosages</label>
                  <input
                    id="current_medication"
                    type="text"
                    name="current_medication"
                    value={form.current_medication}
                    onChange={handleChange}
                    placeholder="e.g. Lipitor 10mg, Beta-blockers, none"
                  />
                </div>
              </div>
            )}

            {/* TAB 3: Vitals & ECG Intervals */}
            {activeTab === 'vitals' && (
              <div className="form-section animate-fadeIn">
                <h3 className="form-section-title">Physiological Vitals & ECG Measurements</h3>
                <div className="form-grid">
                  <FormField label="Heart Rate (bpm)" name="heart_rate" type="number" value={form.heart_rate} onChange={handleChange} error={errors.heart_rate} placeholder="30–220" required />
                  <FormField label="Systolic BP (mmHg)" name="systolic_bp" type="number" value={form.systolic_bp} onChange={handleChange} error={errors.systolic_bp} placeholder="e.g. 120" required />
                  <FormField label="Diastolic BP (mmHg)" name="diastolic_bp" type="number" value={form.diastolic_bp} onChange={handleChange} error={errors.diastolic_bp} placeholder="e.g. 80" required />
                  <FormField label="RR Interval (ms)" name="rr_interval" type="number" value={form.rr_interval} onChange={handleChange} error={errors.rr_interval} placeholder="e.g. 880" required />
                  <FormField label="PP Interval (ms)" name="pp_interval" type="number" value={form.pp_interval} onChange={handleChange} error={errors.pp_interval} placeholder="e.g. 875" required />
                  <FormField label="QT Interval (ms)" name="qt_interval" type="number" value={form.qt_interval} onChange={handleChange} error={errors.qt_interval} placeholder="e.g. 390" required />
                  <FormField label="QTc Interval (ms)" name="qtc_interval" type="number" value={form.qtc_interval} onChange={handleChange} error={errors.qtc_interval} placeholder="e.g. 415" required />
                  <FormField label="QRS Duration (ms)" name="qrs_duration" type="number" value={form.qrs_duration} onChange={handleChange} error={errors.qrs_duration} placeholder="e.g. 98" required />
                  <FormField label="PQ Interval (ms)" name="pq_interval" type="number" value={form.pq_interval} onChange={handleChange} error={errors.pq_interval} placeholder="e.g. 155" required />
                </div>

                <h3 className="form-section-title" style={{ borderTop: '1px solid var(--border)', paddingTop: '1.5rem', marginTop: '1.5rem' }}>
                  ⚠️ Dynamic Risk Symptoms during Exercise/Gym
                </h3>
                <div className="checkbox-list" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '0.5rem' }}>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="syncope" checked={form.syncope} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Syncope / Unexplained Fainting</strong></div>
                  </label>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="previous_collapse_during_sports" checked={form.previous_collapse_during_sports} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Previous Collapse during Exercise</strong></div>
                  </label>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="chest_pain_during_exercise" checked={form.chest_pain_during_exercise} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Chest Pain during Exercise</strong></div>
                  </label>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="loss_of_consciousness_during_exercise" checked={form.loss_of_consciousness_during_exercise} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Loss of Consciousness during Exercise</strong></div>
                  </label>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="shortness_of_breath_during_exercise" checked={form.shortness_of_breath_during_exercise} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Shortness of Breath during Exercise</strong></div>
                  </label>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="dizziness_during_exercise" checked={form.dizziness_during_exercise} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Dizziness / Lightheadedness during Exercise</strong></div>
                  </label>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="palpitations_during_exercise" checked={form.palpitations_during_exercise} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Heart Palpitations during Exercise</strong></div>
                  </label>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                    <input type="checkbox" name="fatigue_during_exercise" checked={form.fatigue_during_exercise} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                    <div><strong>Excessive Fatigue during Exercise</strong></div>
                  </label>
                  
                  {/* Gym specific risk parameters */}
                  {form.person_type === 'gym' && (
                    <>
                      <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                        <input type="checkbox" name="heavy_weight_training" checked={form.heavy_weight_training} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                        <div><strong>Heavy Weight Training</strong></div>
                      </label>
                      <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                        <input type="checkbox" name="steroid_usage" checked={form.steroid_usage} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                        <div style={{ color: 'var(--risk-critical)' }}><strong>Anabolic Steroid Usage ⚠️</strong></div>
                      </label>
                      <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                        <input type="checkbox" name="pre_workout_usage" checked={form.pre_workout_usage} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                        <div><strong>High-Stimulant Pre-Workout Usage</strong></div>
                      </label>
                      <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                        <input type="checkbox" name="energy_drink_consumption" checked={form.energy_drink_consumption} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                        <div><strong>Excessive Energy Drink Consumption</strong></div>
                      </label>
                      <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                        <input type="checkbox" name="dehydration_episodes" checked={form.dehydration_episodes} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                        <div><strong>Frequent Dehydration Episodes</strong></div>
                      </label>
                      <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                        <input type="checkbox" name="overtraining_symptoms" checked={form.overtraining_symptoms} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                        <div><strong>Overtraining Symptoms</strong></div>
                      </label>
                      <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                        <input type="checkbox" name="recent_fainting_episodes" checked={form.recent_fainting_episodes} onChange={handleChange} style={{ width: '1.25rem', height: '1.25rem' }} />
                        <div><strong>Recent Fainting Episodes in Gym</strong></div>
                      </label>
                    </>
                  )}
                </div>
              </div>
            )}

            {apiError && (
              <div className="alert danger" style={{ marginTop: '1.5rem' }}>
                <span className="alert-icon">⚠️</span>
                <div>
                  <strong>Prediction failed</strong>
                  <br />
                  {apiError}
                </div>
              </div>
            )}

            <div className="form-actions" style={{ marginTop: '2rem' }}>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => fillSample('sports')}
                  disabled={loading}
                >
                  Fill Sample Athlete
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => fillSample('gym')}
                  disabled={loading}
                >
                  Fill Sample Gym Goer
                </button>
              </div>
              
              <button type="submit" className="btn btn-primary btn-lg" disabled={loading}>
                {loading ? (
                  <>
                    <span className="spinner spinner-sm" /> Running Classifier…
                  </>
                ) : (
                  '⚡ Predict Risk Score'
                )}
              </button>
            </div>
          </form>

          <aside className="form-sidebar">
            <div className="card info-card">
              <h3>Scientific Features</h3>
              <ul className="info-list">
                <li><strong>Sport Type</strong> - Athlete sports category shifts base cardiac expectations.</li>
                <li><strong>ECG Intervals</strong> - Depolarization and repolarization values (QTc, RR, PP).</li>
                <li><strong>Dynamic Profiling</strong> - Custom screening elements for sports vs. heavy weightlifting and gym routines.</li>
              </ul>
            </div>

            <div className="card info-card">
              <h3>Diagnostic Guidelines</h3>
              <div className="risk-legend">
                <div className="legend-item normal">
                  <span>0–40%</span> Low Risk
                </div>
                <div className="legend-item medium">
                  <span>41–70%</span> Moderate Risk
                </div>
                <div className="legend-item critical">
                  <span>71–100%</span> High Risk / Alert
                </div>
              </div>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}

function FormField({
  label,
  name,
  value,
  onChange,
  error,
  type = 'text',
  placeholder,
  required,
  min,
  max,
  step,
}) {
  return (
    <div className="form-field">
      <label htmlFor={name}>
        {label}
        {required && <span className="required">*</span>}
      </label>
      <input
        id={name}
        name={name}
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        className={error ? 'input-error' : ''}
        min={min}
        max={max}
        step={step}
      />
      {error && <span className="field-error">{error}</span>}
    </div>
  );
}
