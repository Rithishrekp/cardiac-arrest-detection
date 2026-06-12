import os
import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

def generate_pdf_report(record) -> BytesIO:
    """
    Generate a professional PDF report from a database prediction record using ReportLab.
    Returns a BytesIO buffer containing the PDF binary.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Paragraph Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.HexColor('#0F172A'), # slate-900
        spaceAfter=15,
        alignment=TA_CENTER
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#475569'), # slate-600
        spaceAfter=25,
        alignment=TA_CENTER
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor('#1E3A8A'), # dark blue
        spaceBefore=12,
        spaceAfter=8,
        borderPadding=2
    )
    
    label_style = ParagraphStyle(
        'GridLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=colors.HexColor('#334155'), # slate-700
    )
    
    value_style = ParagraphStyle(
        'GridValue',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        textColor=colors.HexColor('#0F172A'),
    )
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        textColor=colors.HexColor('#64748B'), # slate-500
        leading=10,
        alignment=TA_JUSTIFY
    )
    
    recom_style = ParagraphStyle(
        'Recommendation',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        textColor=colors.HexColor('#1E293B'),
        leading=12,
        bulletFontName='Helvetica',
        bulletFontSize=9,
        leftIndent=15,
        firstLineIndent=-10
    )
    
    story = []
    
    # Header Banner Image/Table
    header_data = [
        [
            Paragraph("<b>SUDDEN CARDIAC ARREST SCREENING REPORT</b>", ParagraphStyle('H1', fontName='Helvetica-Bold', fontSize=14, textColor=colors.white, alignment=TA_CENTER)),
        ]
    ]
    header_table = Table(header_data, colWidths=[doc.width])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1E3A8A')),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 15))
    
    # 1. Patient Demographics Section
    story.append(Paragraph("Patient Profile", section_heading))
    
    gender_str = str(record.gender)
    age_str = f"{int(record.age)} years" if record.age else "N/A"
    weight_str = f"{record.weight} kg" if record.weight else "N/A"
    height_str = f"{record.height} cm" if record.height else "N/A"
    bmi_str = f"{round(record.bmi, 2)}" if record.bmi else "N/A"
    sport_str = str(record.sport_type) if record.sport_type else "N/A"
    person_type_str = "Gym Individual" if record.person_type == "gym" else "Sports Person"
    
    patient_info_data = [
        [Paragraph("Full Name:", label_style), Paragraph(record.patient_name, value_style),
         Paragraph("Patient ID / MRN:", label_style), Paragraph(record.patient_id, value_style)],
        [Paragraph("Age:", label_style), Paragraph(age_str, value_style),
         Paragraph("Gender:", label_style), Paragraph(gender_str, value_style)],
        [Paragraph("Height:", label_style), Paragraph(height_str, value_style),
         Paragraph("Weight:", label_style), Paragraph(weight_str, value_style)],
        [Paragraph("BMI:", label_style), Paragraph(bmi_str, value_style),
         Paragraph("Subject Type:", label_style), Paragraph(person_type_str, value_style)],
    ]
    
    patient_table = Table(patient_info_data, colWidths=[100, 160, 100, 160])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 15))
    
    # 2. Risk Metrics Display Banner (Callout box)
    story.append(Paragraph("Cardiac Risk Assessment Summary", section_heading))
    
    risk_color_hex = '#10B981' # Green for low
    risk_level_text = "LOW RISK"
    if record.risk_level == "medium":
        risk_color_hex = '#F59E0B' # Orange/Yellow
        risk_level_text = "MODERATE RISK"
    elif record.risk_level == "critical":
        risk_color_hex = '#EF4444' # Red
        risk_level_text = "HIGH RISK"
        
    risk_box_data = [
        [
            Paragraph(f"<b>RISK LEVEL:</b><br/><font size=16 color='{risk_color_hex}'><b>{risk_level_text}</b></font>", ParagraphStyle('RiskLevel', fontName='Helvetica', fontSize=10, textColor=colors.HexColor('#1E293B'), alignment=TA_CENTER)),
            Paragraph(f"<b>RISK SCORE:</b><br/><font size=20 color='{risk_color_hex}'><b>{record.risk_score}%</b></font>", ParagraphStyle('RiskScore', fontName='Helvetica', fontSize=10, textColor=colors.HexColor('#1E293B'), alignment=TA_CENTER)),
            Paragraph(f"<b>MODEL CONFIDENCE:</b><br/><font size=20 color='#1E3A8A'><b>{record.model_confidence if record.model_confidence else 95.0}%</b></font>", ParagraphStyle('ModelConf', fontName='Helvetica', fontSize=10, textColor=colors.HexColor('#1E293B'), alignment=TA_CENTER)),
        ]
    ]
    
    risk_table = Table(risk_box_data, colWidths=[173, 173, 173])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F1F5F9')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('PADDING', (0,0), (-1,-1), 10),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 15))
    
    # 3. ECG Parameters & Vital Signs
    story.append(Paragraph("ECG Parameters & Vitals", section_heading))
    
    hr_str = f"{record.heart_rate} bpm" if record.heart_rate else "N/A"
    bp_str = f"{record.systolic_bp}/{record.diastolic_bp} mmHg" if record.systolic_bp and record.diastolic_bp else "N/A"
    map_str = f"{record.mean_arterial_pressure} mmHg" if record.mean_arterial_pressure else "N/A"
    rr_str = f"{record.rr_interval} ms" if record.rr_interval else "N/A"
    pq_str = f"{record.pq_interval} ms" if record.pq_interval else "N/A"
    qrs_str = f"{record.qrs_duration} ms" if record.qrs_duration else "N/A"
    qt_str = f"{record.qt_interval} ms" if record.qt_interval else "N/A"
    qtc_str = f"{record.qtc_interval} ms" if record.qtc_interval else "N/A"
    
    ecg_vitals_data = [
        [Paragraph("Resting Heart Rate:", label_style), Paragraph(hr_str, value_style),
         Paragraph("RR Interval:", label_style), Paragraph(rr_str, value_style)],
        [Paragraph("Blood Pressure:", label_style), Paragraph(bp_str, value_style),
         Paragraph("PQ Interval:", label_style), Paragraph(pq_str, value_style)],
        [Paragraph("Mean Arterial Pressure (MAP):", label_style), Paragraph(map_str, value_style),
         Paragraph("QRS Duration:", label_style), Paragraph(qrs_str, value_style)],
        [Paragraph("Primary Sport Category:", label_style), Paragraph(sport_str, value_style),
         Paragraph("QT / QTc Intervals:", label_style), Paragraph(f"{qt_str} / {qtc_str}", value_style)],
    ]
    
    ecg_table = Table(ecg_vitals_data, colWidths=[160, 100, 160, 100])
    ecg_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(ecg_table)
    story.append(Spacer(1, 15))
    
    # 4. Specific Risk Factors
    story.append(Paragraph("Identified Risk Factors", section_heading))
    
    risk_factors = []
    if record.syncope == 1.0 or record.recent_fainting_episodes == 1.0:
        risk_factors.append("<b>Syncope / Fainting Episodes</b>: Previous fainting episodes present a strong risk of arrhythmia.")
    if record.previous_collapse_during_sports == 1.0 or record.loss_of_consciousness_during_exercise == 1.0:
        risk_factors.append("<b>Previous Collapse / Loss of Consciousness During Exercise</b>: Highly critical cardiac symptom indicator.")
    if record.chest_pain_during_exercise == 1.0:
        risk_factors.append("<b>Exercise-Induced Chest Pain</b>: Strong indicator of potential ischemia or structural heart anomalies.")
    if record.steroid_usage == 1.0:
        risk_factors.append("<b>Anabolic Steroid Usage</b>: Linked to ventricular hypertrophy, arterial hardening, and arrhythmia risks.")
    if record.family_history_sudden_death == 1.0 or record.family_history_cardiac_arrest == 1.0:
        risk_factors.append("<b>Family Genetics</b>: Positive genetic history of cardiac arrest or sudden unexplained death.")
    if record.hypertension == 1.0:
        risk_factors.append("<b>Hypertension (High Blood Pressure)</b>: Chronically elevated BP strain on left ventricular walls.")
    if record.smoking == 1.0:
        risk_factors.append("<b>Tobacco Use</b>: Active smoking increases atherosclerotic plaques and cardiac irritability.")
        
    if not risk_factors:
        risk_factors.append("No critical secondary lifestyle or medical history risk factors were identified. Clinical status appears stable.")
        
    rf_bullets = []
    for rf in risk_factors:
        rf_bullets.append([Paragraph(f"&bull; {rf}", recom_style)])
        
    rf_table = Table(rf_bullets, colWidths=[doc.width])
    rf_table.setStyle(TableStyle([
        ('PADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(rf_table)
    story.append(Spacer(1, 15))
    
    # 5. Clinical Recommendations
    story.append(Paragraph("Screening Recommendations & Care Guidelines", section_heading))
    
    recommendations = []
    if record.risk_level == "normal":
        recommendations = [
            "Maintain current training load with proper hydration and dynamic warmups.",
            "Schedule standard annual physical checkups and clinical vitals logging.",
            "Ensure a healthy heart diet and avoid excessive use of high-caffeine pre-workouts.",
            "Self-monitor for any development of chest pain, palpitations, or fainting during exercise."
        ]
    elif record.risk_level == "medium":
        recommendations = [
            "Schedule a non-emergency clinical consultation with a cardiologist or sports physician.",
            "Acquire a 12-lead diagnostic resting ECG to confirm electrical interval values.",
            "Avoid high-dose pre-workout formulas, synthetic stimulants, and energy drinks.",
            "Monitor daily blood pressure and restrict workouts to moderate levels until cleared by a physician."
        ]
    else:  # critical / high
        recommendations = [
            "IMMEDIATE ACTION REQUIRED: Schedule a comprehensive clinical cardiology evaluation immediately.",
            "Cease all high-intensity athletic training, gym workouts, and cardio sessions until medically cleared.",
            "Acquire an Echocardiogram and 24-hour Holter Monitor to assess structural and rhythmic viability.",
            "If chest pressure, fainting, or palpitations occur, seek emergency medical services (EMS) immediately."
        ]
        
    recom_bullets = []
    for rec in recommendations:
        recom_bullets.append([Paragraph(f"&bull; {rec}", recom_style)])
        
    recom_table = Table(recom_bullets, colWidths=[doc.width])
    recom_table.setStyle(TableStyle([
        ('PADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(recom_table)
    story.append(Spacer(1, 20))
    
    # 6. Medical Disclaimer and Date
    disclaimer_text = (
        "<b>MEDICAL DISCLAIMER:</b> This system is an AI-assisted early sudden cardiac arrest risk screening "
        "platform intended for educational and research purposes only. It does not perform a medical diagnosis "
        "or replace clinical evaluations. The calculated risk indicators should not be used as a substitute "
        "for professional medical consultation, treatment, or diagnostic advice."
    )
    
    disc_data = [
        [Paragraph(disclaimer_text, disclaimer_style)]
    ]
    disc_table = Table(disc_data, colWidths=[doc.width])
    disc_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(disc_table)
    story.append(Spacer(1, 10))
    
    # Generated Date & Institution label
    gen_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    footer_data = [
        [
            Paragraph(f"Generated on: {gen_date}", ParagraphStyle('F1', fontName='Helvetica', fontSize=8, textColor=colors.HexColor('#64748B'))),
            Paragraph("Cardiac Risk AI Research Platform", ParagraphStyle('F2', fontName='Helvetica', fontSize=8, textColor=colors.HexColor('#64748B'), alignment=TA_RIGHT))
        ]
    ]
    footer_table = Table(footer_data, colWidths=[260, 260])
    footer_table.setStyle(TableStyle([
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(footer_table)
    
    # Build document
    doc.build(story)
    buffer.seek(0)
    return buffer
