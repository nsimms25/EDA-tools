"""
THIS IS THE OUTLINE OF THE PROJECT FOR HYPOTHESIS TESTING. 

Create a toolkit or web app that:
Accepts two or more groups or categorical data as input.
Automatically selects and performs the correct statistical test.
Outputs p-values, test statistics, confidence intervals, and a decision on the null hypothesis.
Optionally generates visualizations and a PDF/HTML report.

Parametric vs non-parametric tests
Types of errors (Type I, II)
Assumptions of tests
Inference interpretation (p-value, effect size)
Choosing the correct test for the data

Supported Statistical Tests
Scenario	Suggested Test
Compare means of 2 groups	Independent t-test
Compare means of >2 groups	One-way ANOVA
Before/after in same group	Paired t-test
Association between categorical vars	Chi-squared test of independence
Compare proportions between 2 groups	Z-test for proportions
Non-normal 2-group comparison	Mann-Whitney U test

Python stack:
    pandas, scipy.stats, statsmodels, numpy
    Optional: seaborn, matplotlib for plots
    Optional: streamlit for interactive UI
    Optional: jinja2, pdfkit for generating reports

Bootstrap resampling alternative to tests.
Confidence interval visualization.
Auto-check assumptions (e.g., normality with Shapiro-Wilk, Levene’s test).
User-uploaded CSVs via Streamlit.
Report export as PDF or HTML.

"""

