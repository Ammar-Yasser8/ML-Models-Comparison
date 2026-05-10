from html import escape
from pathlib import Path


def save_model_report(
        *,
        dataset_name,
        model_name,
        accuracy,
        precision,
        recall,
        f1_score,
        confusion,
        class_report,
        output_path,
):
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".html":
                output_path = output_path.with_suffix(".html")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset_title = dataset_name.replace("_", " ").title()
        model_title = model_name.upper()
        confusion_text = escape(str(confusion))
        class_report_text = escape(class_report.strip())

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dataset_title} - {model_title} Report</title>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f4f7fb;
            --panel: rgba(255, 255, 255, 0.9);
            --border: rgba(15, 23, 42, 0.08);
            --text: #0f172a;
            --muted: #64748b;
            --accent: #2563eb;
            --accent-soft: rgba(37, 99, 235, 0.08);
            --shadow: 0 22px 54px rgba(15, 23, 42, 0.12);
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.16), transparent 30%),
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.12), transparent 26%),
                linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
            min-height: 100vh;
        }}
        .wrap {{ max-width: 1120px; margin: 0 auto; padding: 40px 20px 60px; }}
        .hero, .panel, .metric {{
            background: var(--panel);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            backdrop-filter: blur(16px);
        }}
        .hero {{ border-radius: 28px; padding: 28px; }}
        .badge {{
            display: inline-flex; align-items: center; gap: 8px;
            padding: 8px 12px; border-radius: 999px;
            background: var(--accent-soft); color: var(--accent);
            font-size: 0.9rem; font-weight: 700;
        }}
        h1 {{ margin: 14px 0 8px; font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.05; }}
        .subtitle {{ margin: 0; color: var(--muted); max-width: 68ch; }}
        .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin-top: 18px; }}
        .metric {{ border-radius: 22px; padding: 18px; }}
        .label {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 10px; }}
        .value {{ font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; color: var(--accent); }}
        .grid {{ display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 16px; margin-top: 18px; }}
        .panel {{ border-radius: 24px; padding: 22px; }}
        .panel h2 {{ margin: 0 0 14px; font-size: 1.15rem; }}
        pre {{
            margin: 0;
            padding: 18px;
            overflow: auto;
            border-radius: 18px;
            background: #0f172a;
            color: #e2e8f0;
            font-size: 0.95rem;
            line-height: 1.5;
        }}
        .summary {{ display: grid; gap: 12px; }}
        .summary-item {{
            display: flex; justify-content: space-between; gap: 16px;
            padding: 14px 16px; border-radius: 16px;
            background: #f8fafc; border: 1px solid #e2e8f0;
        }}
        .summary-item span:first-child {{ color: var(--muted); font-weight: 600; }}
        .summary-item span:last-child {{ font-weight: 800; }}
        @media (max-width: 900px) {{
            .metrics, .grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <main class="wrap">
        <section class="hero">
            <div class="badge">Machine Learning Classification Report</div>
            <h1>{dataset_title} - {model_title}</h1>
            <p class="subtitle">A clean, browser-friendly report showing the key evaluation metrics, confusion matrix, and full classification summary for this model.</p>
        </section>

        <section class="metrics">
            <div class="metric"><div class="label">Accuracy</div><div class="value">{accuracy:.4f}</div></div>
            <div class="metric"><div class="label">Precision</div><div class="value">{precision:.4f}</div></div>
            <div class="metric"><div class="label">Recall</div><div class="value">{recall:.4f}</div></div>
            <div class="metric"><div class="label">F1 Score</div><div class="value">{f1_score:.4f}</div></div>
        </section>

        <section class="grid">
            <article class="panel">
                <h2>Confusion Matrix</h2>
                <pre>{confusion_text}</pre>
            </article>

            <article class="panel">
                <h2>Model Summary</h2>
                <div class="summary">
                    <div class="summary-item"><span>Dataset</span><span>{dataset_title}</span></div>
                    <div class="summary-item"><span>Model</span><span>{model_title}</span></div>
                    <div class="summary-item"><span>Output</span><span>HTML Page</span></div>
                    <div class="summary-item"><span>Style</span><span>Glass / Card Layout</span></div>
                </div>
            </article>
        </section>

        <section class="panel" style="margin-top:16px;">
            <h2>Classification Report</h2>
            <pre>{class_report_text}</pre>
        </section>
    </main>
</body>
</html>
"""

        output_path.write_text(html, encoding="utf-8")