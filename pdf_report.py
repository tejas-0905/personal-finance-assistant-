from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    PageBreak, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
import plotly.io as pio
import os
import io
from PIL import Image as PILImage


def save_plotly(fig, filename):
    """Save plotly figure as image with error handling"""
    try:
        img_bytes = pio.to_image(fig, format="png", engine="kaleido")
        with open(filename, "wb") as f:
            f.write(img_bytes)
        return filename
    except (RuntimeError, Exception) as e:
        # Fallback: create a simple placeholder image or skip the image
        print(f"Warning: Could not save plotly figure to {filename}: {e}")
        # Create a simple placeholder image using reportlab
        from reportlab.lib.utils import ImageReader
        from PIL import Image as PILImage
        import numpy as np

        try:
            # Create a simple placeholder image
            img = PILImage.new('RGB', (400, 200), color=(240, 240, 240))
            img.save(filename)
            return filename
        except:
            # If even PIL fails, return None to skip the image
            return None


def generate_pdf_report(data):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    # ---------------- CUSTOM STYLES ----------------
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        alignment=1,
        textColor=colors.HexColor("#6366f1"),
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        alignment=1,
        textColor=colors.grey,
        fontSize=10,
        spaceAfter=20
    )

    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        textColor=colors.HexColor("#4f46e5"),
        spaceAfter=12
    )

    card_style = ParagraphStyle(
        'Card',
        parent=styles['BodyText'],
        textColor=colors.white,
        alignment=1
    )

    normal_style = ParagraphStyle(
        'NormalCustom',
        parent=styles['BodyText'],
        spaceAfter=8
    )

    content = []

    # ---------------- HEADER ----------------
    content.append(Paragraph("💰 Personal Finance Report", title_style))
    content.append(Paragraph("AI-powered financial insights & analytics", subtitle_style))

    # ---------------- KPI CARDS ----------------
    kpi_data = [
        [
            Paragraph(f"<b>Total Spend</b><br/>₹{data['total_spend']:,.0f}", styles['BodyText']),
            Paragraph(f"<b>Total Income</b><br/>₹{data['total_income']:,.0f}", styles['BodyText'])
        ],
        [
            Paragraph(f"<b>Net Cashflow</b><br/>₹{data['net_cashflow']:,.0f}", styles['BodyText']),
            Paragraph(f"<b>Savings Rate</b><br/>{data['savings_rate']:.1f}%", styles['BodyText'])
        ]
    ]

    table = Table(kpi_data, colWidths=[250, 250])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#cdd8ea")),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.white),
        ("BOX", (0,0), (-1,-1), 1, colors.HexColor("#6366f1")),
        ("INNERGRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTSIZE", (0,0), (-1,-1), 12),
        ("BOTTOMPADDING", (0,0), (-1,-1), 15),
        ("TOPPADDING", (0,0), (-1,-1), 15),
    ]))

    content.append(table)
    content.append(Spacer(1, 25))

    # ---------------- CHARTS ----------------
    content.append(Paragraph("📊 Spending Analysis", section_style))

    image_paths = []
    chart_pairs = [data["charts"][i:i+2] for i in range(0, len(data["charts"]), 2)]
    for pair in chart_pairs:
        row = []


        for title, fig in pair:
            img_path = f"{title}.png".replace(" ", "_")
            saved_path = save_plotly(fig, img_path)

            if saved_path:  # Only add image if saving was successful
                image_paths.append(img_path)

                img = Image(img_path)
                img.drawHeight = 200   # ✅ optimized for A4
                img.drawWidth = 240    # ✅ half width

                block = [
                    Paragraph(title, styles['Heading4']),
                    Spacer(1, 5),
                    img
                ]
            else:
                # Fallback: just add the title without image
                block = [
                    Paragraph(f"{title} (Chart not available)", styles['Heading4']),
                    Spacer(1, 5),
                    Paragraph("Chart generation failed", styles['Normal'])
                ]

            row.append(block)

        # Only create table if row has content
        if row:
            # If only one chart, add empty placeholder
            if len(row) == 1:
                row.append("")
            table = Table([row], colWidths=[260, 260])

            table.setStyle(TableStyle([
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("ALIGN", (0,0), (-1,-1), "CENTER"),
                ("LEFTPADDING", (0,0), (-1,-1), 10),
                ("RIGHTPADDING", (0,0), (-1,-1), 10),
                ("BOTTOMPADDING", (0,0), (-1,-1), 20),
            ]))

            content.append(table)
            content.append(Spacer(1, 15))

            # Force page break after every 2 rows (i.e., 4 charts per page max)
            if chart_pairs.index(pair) % 2 == 1:
                content.append(PageBreak())
        

    # ---------------- FORECAST ----------------
    if data.get("forecast_fig"):
        img_path = "forecast.png"
        saved_path = save_plotly(data["forecast_fig"], img_path)

        content.append(Paragraph("📈 Expense Forecast", section_style))

        if saved_path:  # Only add image if saving was successful
            image_paths.append(img_path)

            img = Image(img_path)
            img.drawHeight = 260
            img.drawWidth = 460

            content.append(img)
        else:
            content.append(Paragraph("Forecast chart generation failed", styles['Normal']))

        content.append(Spacer(1, 20))

    # ---------------- RECOMMENDATIONS ----------------
    content.append(Paragraph("💡 Smart Recommendations", section_style))

    for rec in data["recommendations"]:
        box = Table([[f"• {rec}"]], colWidths=[500])
        box.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0,0), (-1,-1), colors.white),
            ("BOX", (0,0), (-1,-1), 1, colors.HexColor("#6366f1")),
            ("LEFTPADDING", (0,0), (-1,-1), 10),
            ("RIGHTPADDING", (0,0), (-1,-1), 10),
            ("TOPPADDING", (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ]))

        content.append(box)
        content.append(Spacer(1, 10))

    # ---------------- FOOTER ----------------
    content.append(Spacer(1, 30))
    content.append(Paragraph(
        "Generated by MoneyMentor • AI Personal Finance Assistant",
        subtitle_style
    ))

    # Build PDF
    doc.build(content)

    # Cleanup
    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)

    buffer.seek(0)
    return buffer