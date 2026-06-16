"""
Comprehensive PDF report for the Africa undernourishment ML + statistics project,
with interpretation and full code (appendices).
Run:  python 03_make_report.py  ->  report/Africa_Nutrition_ML_Report.pdf
"""
import json
from pathlib import Path
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                TableStyle, PageBreak, Preformatted, ListFlowable, ListItem)

FIG = Path("outputs"); OUT = Path("report"); OUT.mkdir(exist_ok=True)
m = json.load(open(FIG / "metrics.json"))
A = "Kingsley Amegah"
INK = colors.HexColor("#16212e"); ACCENT = colors.HexColor("#d73027"); MUTED = colors.HexColor("#5b6573")
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10.5, leading=15.5, alignment=TA_JUSTIFY, spaceAfter=8, textColor=colors.HexColor("#222a31"))
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, leading=19, textColor=INK, spaceBefore=14, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, leading=16, textColor=ACCENT, spaceBefore=10, spaceAfter=4)
cap = ParagraphStyle("c", parent=ss["BodyText"], fontSize=8.5, leading=11, textColor=MUTED, alignment=TA_CENTER, spaceBefore=3, spaceAfter=12)
codest = ParagraphStyle("code", parent=ss["BodyText"], fontName="Courier", fontSize=7, leading=9, textColor=INK,
                        backColor=colors.HexColor("#f4f5f7"), borderColor=colors.HexColor("#e0e3e8"),
                        borderWidth=0.5, borderPadding=5, spaceAfter=12)
def P(t): return Paragraph(t, body)
def H1(t): return Paragraph(t, h1)
def H2(t): return Paragraph(t, h2)
def bl(items): return ListFlowable([ListItem(Paragraph(t, body), value="•") for t in items], bulletType="bullet", leftIndent=14, spaceAfter=8)
def fig(path, c, mh=4.2*inch):
    iw, ih = PILImage.open(path).size; w = 6.6*inch; h = w*ih/iw
    if h > mh: h = mh; w = h*iw/ih
    im = Image(str(path), width=w, height=h); im.hAlign = "CENTER"
    return [im, Paragraph(c, cap)]

s = []
s += [Spacer(1, 1.4*inch),
  Paragraph("MACHINE LEARNING + STATISTICS REPORT", ParagraphStyle("k", parent=body, alignment=TA_CENTER, textColor=ACCENT, fontSize=11)),
  Spacer(1, .15*inch),
  Paragraph("Predicting Undernourishment in Africa", ParagraphStyle("t", parent=ss["Title"], fontSize=25, leading=31, textColor=INK)),
  Spacer(1, .1*inch),
  Paragraph("Ten machine-learning algorithms, validated honestly, and supported by statistical models",
            ParagraphStyle("su", parent=ss["Title"], fontSize=13, leading=18, textColor=MUTED, fontName="Helvetica")),
  Spacer(1, 1.5*inch),
  Paragraph(f"<b>{A}</b>", ParagraphStyle("a", parent=ss["Title"], fontSize=15, textColor=INK)),
  Paragraph("Health Data Scientist", ParagraphStyle("r", parent=ss["Title"], fontSize=11, textColor=MUTED, fontName="Helvetica")),
  Spacer(1, .3*inch),
  Paragraph("Python &middot; scikit-learn &middot; statsmodels &middot; SHAP &nbsp;|&nbsp; World Bank Open Data &nbsp;|&nbsp; github.com/Kingsley-amg/africa-nutrition-ml",
            ParagraphStyle("f", parent=ss["Title"], fontSize=9, textColor=MUTED, fontName="Helvetica")),
  PageBreak()]

s.append(H1("Executive summary"))
s.append(P(f"This project predicts <b>undernourishment</b> (the prevalence of food "
           f"insecurity) across <b>{m['n_countries']} African countries</b> over "
           f"{m['year_range'][0]}-{m['year_range'][1]} ({m['n_rows']:,} country-year "
           "observations), using World Bank data. It compares ten machine-learning "
           "algorithms and supports the findings with formal statistical models."))
s.append(bl([
  f"<b>The headline is a lesson in honest evaluation.</b> Under ordinary random "
  f"cross-validation the best model reaches R-squared of {m['best_r2_random']:.2f}, "
  "which looks superb - but it is misleading, because the same country appears in "
  "both training and test data.",
  f"Under <b>country-held-out validation</b>, the best model ({m['best_model']}) "
  f"reaches only R-squared {m['best_r2_grouped']:.2f}: predicting undernourishment "
  "for a country never seen in training is genuinely hard.",
  "<b>Income is the dominant correlate</b> of undernourishment, followed by water "
  "access, sanitation and health spending; higher fertility and agricultural "
  "employment go with more undernourishment.",
  "These directions are <b>consistent across correlation, OLS regression and a "
  "mixed-effects panel model</b>, which together explain why prediction is hard: "
  "much of the variation is persistent and country-specific."]))

s.append(H1("1. Introduction and question"))
s.append(P("Undernourishment remains a defining development challenge across much of "
           "Africa. This project asks two linked questions: <b>how well can we predict</b> "
           "a country's undernourishment from its socio-economic, agricultural and health "
           "conditions, and <b>which factors are most associated</b> with it? The first is "
           "a machine-learning question, the second a statistical one, and the project "
           "treats them together."))

s.append(H1("2. Data and methods"))
s.append(P(f"Data come from the <b>World Bank Open Data API</b>: the prevalence of "
           "undernourishment as the outcome, plus twelve determinants (GDP per capita, "
           "health spending, water and sanitation access, fertility, rural population "
           "share, food production, agricultural employment and value added, population "
           "growth, primary enrolment and inflation). GDP and health spending were "
           "log-transformed. Ten regression algorithms were compared under two "
           "cross-validation designs - ordinary random K-fold and <b>GroupKFold grouped "
           "by country</b> - and the findings were checked with correlation analysis, an "
           "OLS regression with full inference, and a mixed-effects panel model. Full "
           "code is in Appendices A and B."))

s.append(H1("3. Exploratory analysis"))
s.append(P("Undernourishment has fallen on average but remains high and uneven across "
           "African regions (Figure 1). The strongest bivariate correlates are water "
           "access, income and sanitation (all negative) and fertility (positive)."))
s += fig(FIG / "01_trend_by_region.png", "Figure 1. Mean undernourishment over time, by African region.", 3.0*inch)
s += fig(FIG / "02_correlations.png", "Figure 2. Correlation of each determinant with undernourishment.", 3.2*inch)

s.append(H1("4. Machine learning: ten algorithms, two validation designs"))
lb = m["leaderboard"]
tbl = Table([["Algorithm", "Random-CV R2", "Grouped-CV R2", "RMSE", "MAE"]] +
            [[r["model"], f"{r['R2_random']:.2f}", f"{r['R2_grouped']:.2f}",
              f"{r['RMSE_grouped']:.1f}", f"{r['MAE_grouped']:.1f}"] for r in lb],
            colWidths=[2.2*inch, 1.2*inch, 1.2*inch, 0.9*inch, 0.9*inch])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0),(-1,0), INK), ("TEXTCOLOR",(0,0),(-1,0), colors.white),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),8.5),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f3f5f7")]),
    ("GRID",(0,0),(-1,-1),0.4, colors.HexColor("#d6dbe0")),
    ("ALIGN",(1,0),(-1,-1),"CENTER"), ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
s += [tbl, Spacer(1, 8)]
s.append(P(f"The contrast between the two columns is the most important result. Random "
           f"cross-validation rewards models that memorise each country's level from its "
           f"other years, inflating R-squared to {m['best_r2_random']:.2f}. The honest "
           f"test - holding out whole countries - collapses this to {m['best_r2_grouped']:.2f} "
           "for the best model, and to around zero or below for the linear and simpler "
           "models. Tree ensembles (Extra Trees, Random Forest) generalise best, but even "
           "they explain only a modest share of variation in unseen countries."))
s += fig(FIG / "03_ml_leaderboard.png", "Figure 3. Ten algorithms under random vs. country-grouped cross-validation.", 3.4*inch)
s += fig(FIG / "04_pred_vs_actual.png", "Figure 4. Best model: predicted vs. actual undernourishment for held-out countries.", 3.2*inch)

s.append(H1("5. Explainability (SHAP)"))
s.append(P("SHAP values for the random-forest model confirm the statistical picture: "
           "low income, low water and sanitation access, and high fertility push predicted "
           "undernourishment up. The agreement between a flexible ML model and the "
           "regression models is reassuring - the signal is real, not an artefact of one "
           "method."))
s += fig(FIG / "05_shap.png", "Figure 5. SHAP summary: how each determinant moves predicted undernourishment.", 4.2*inch)

s.append(H1("6. Statistical models"))
s.append(P(f"A standardised OLS regression explains {m['ols_r2']*100:.0f}% of the "
           "in-sample variation. Coefficients are reported with 95% confidence intervals "
           "(Figure 6); income has by far the largest standardised effect. Multicollinearity "
           f"is moderate (max VIF = {m['max_vif']}), so individual coefficients - "
           "particularly for the correlated income and spending variables - should be read "
           "with care. Because the data is a country panel with repeated measures, a "
           "<b>mixed-effects model with a random intercept per country</b> was also fitted; "
           "it confirms income as a strong negative correlate while absorbing persistent "
           "country differences, which is precisely why purely cross-country prediction is "
           "limited."))
s += fig(FIG / "06_ols_coefficients.png", "Figure 6. Standardised OLS coefficients with 95% confidence intervals.", 3.2*inch)

s.append(H1("7. Conclusion"))
s.append(P("Across ten algorithms and three statistical models, the determinants of "
           "undernourishment in Africa are consistent and intuitive: income, clean water, "
           "sanitation and lower fertility are associated with less food insecurity. Yet "
           "the project's most valuable lesson is methodological. A model that looks near-"
           "perfect under careless validation is shown, under honest country-held-out "
           "testing, to generalise only modestly - because undernourishment is deeply "
           "country-specific. Reporting both is what separates a trustworthy analysis from "
           "an over-optimistic one."))

s.append(H1("8. Limitations"))
s.append(bl([
  "Country-level (ecological) analysis; relationships need not hold for individuals.",
  "Observational data: associations, not causal effects.",
  "Some determinants are collinear (max VIF near 7), affecting individual OLS coefficients.",
  "Undernourishment is partly modelled by the World Bank and updated on a lag.",
  "Survey-based predictors carry measurement error and uneven coverage."]))

s.append(PageBreak()); s.append(H1("Appendix A - Data extraction (01_extract.py)"))
s.append(Preformatted(Path("01_extract.py").read_text(), codest))
s.append(PageBreak()); s.append(H1("Appendix B - Analysis (02_analyze.py)"))
s.append(Preformatted(Path("02_analyze.py").read_text(), codest))

def footer(c, d):
    c.saveState(); c.setStrokeColor(colors.HexColor("#d6dbe0")); c.setLineWidth(.5)
    c.line(.8*inch, .7*inch, letter[0]-.8*inch, .7*inch)
    c.setFont("Helvetica", 8); c.setFillColor(MUTED)
    c.drawString(.8*inch, .55*inch, f"Predicting Undernourishment in Africa  |  {A}")
    c.drawRightString(letter[0]-.8*inch, .55*inch, f"Page {d.page}"); c.restoreState()

doc = SimpleDocTemplate(str(OUT / "Africa_Nutrition_ML_Report.pdf"), pagesize=letter,
    leftMargin=.8*inch, rightMargin=.8*inch, topMargin=.8*inch, bottomMargin=.9*inch,
    title="Predicting Undernourishment in Africa", author=A,
    subject="Machine learning and statistics report", creator=A)
doc.build(s, onLaterPages=footer)
print("Wrote", OUT / "Africa_Nutrition_ML_Report.pdf")
