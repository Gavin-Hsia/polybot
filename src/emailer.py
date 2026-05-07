"""
Sends a daily picks email via Gmail SMTP.
Set GMAIL_ADDRESS and GMAIL_APP_PASSWORD in your .env file.
"""

import os
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

GMAIL_ADDRESS  = os.getenv("GMAIL_ADDRESS")
GMAIL_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
TO_ADDRESS     = os.getenv("NOTIFY_EMAIL", GMAIL_ADDRESS)


def _bet_row_html(match: dict, bet: dict, rank: int) -> str:
    tag        = "★ UPSET PICK" if bet["is_upset_pick"] else "Value Pick"
    tag_color  = "#c0392b" if bet["is_upset_pick"] else "#2980b9"
    edge_color = "#27ae60" if bet["edge"] >= 0.10 else "#f39c12"
    return f"""
    <tr>
      <td style="padding:10px 8px;border-bottom:1px solid #eee;font-weight:bold;color:#333;">
        #{rank} &nbsp;<span style="color:{tag_color};font-size:12px;">{tag}</span>
      </td>
      <td style="padding:10px 8px;border-bottom:1px solid #eee;">
        <strong>{match['title']}</strong><br>
        <span style="color:#555;font-size:13px;">{bet['surface']} | Closes {match['close_time'][:16]} UTC</span>
      </td>
      <td style="padding:10px 8px;border-bottom:1px solid #eee;">
        YES on <strong>{bet['player']}</strong><br>
        <span style="color:#888;font-size:13px;">vs {bet['opponent']}</span>
      </td>
      <td style="padding:10px 8px;border-bottom:1px solid #eee;text-align:center;">
        {bet['model_prob']:.1%}
      </td>
      <td style="padding:10px 8px;border-bottom:1px solid #eee;text-align:center;">
        {bet['kalshi_mid']:.2f}
      </td>
      <td style="padding:10px 8px;border-bottom:1px solid #eee;text-align:center;
                 font-weight:bold;color:{edge_color};">
        {bet['edge']:+.1%}
      </td>
      <td style="padding:10px 8px;border-bottom:1px solid #eee;text-align:center;">
        {bet['kelly']:.1%}
      </td>
      <td style="padding:10px 8px;border-bottom:1px solid #eee;text-align:center;color:#888;font-size:13px;">
        ${bet['volume']:,.0f}
      </td>
    </tr>"""


def _build_html(bets: list[tuple]) -> str:
    today     = datetime.now(timezone.utc).strftime("%B %d, %Y")
    n_upset   = sum(1 for _, b in bets if b["is_upset_pick"])
    n_value   = len(bets) - n_upset

    rows_html = ""
    for i, (match, bet) in enumerate(bets, 1):
        rows_html += _bet_row_html(match, bet, i)

    if not rows_html:
        rows_html = """
        <tr><td colspan="8" style="padding:20px;text-align:center;color:#888;">
          No value bets found today.
        </td></tr>"""

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:Arial,sans-serif;background:#f5f5f5;margin:0;padding:20px;">
  <div style="max-width:900px;margin:0 auto;background:#fff;border-radius:8px;
              box-shadow:0 2px 8px rgba(0,0,0,0.1);overflow:hidden;">

    <!-- Header -->
    <div style="background:#1a1a2e;padding:24px 28px;">
      <h1 style="color:#fff;margin:0;font-size:22px;">Polybot · ATP Tennis Picks</h1>
      <p style="color:#aaa;margin:6px 0 0;">{today}</p>
    </div>

    <!-- Summary bar -->
    <div style="background:#f8f9fa;padding:14px 28px;border-bottom:1px solid #eee;
                display:flex;gap:32px;">
      <span style="font-size:14px;color:#555;">
        Total picks: <strong>{len(bets)}</strong>
      </span>
      <span style="font-size:14px;color:#c0392b;">
        ★ Upset picks: <strong>{n_upset}</strong>
      </span>
      <span style="font-size:14px;color:#2980b9;">
        Value picks: <strong>{n_value}</strong>
      </span>
    </div>

    <!-- Key -->
    <div style="padding:12px 28px;background:#fffbf0;border-bottom:1px solid #f0e8c8;
                font-size:13px;color:#666;">
      <strong>How to read:</strong>
      <span style="color:#c0392b;">★ Upset Pick</span> = Kalshi has this player as underdog (&lt;45%) but our model says they should win.
      &nbsp;|&nbsp;
      <strong>Edge</strong> = model probability minus Kalshi ask price — the larger, the better.
      &nbsp;|&nbsp;
      <strong>Kelly</strong> = suggested % of bankroll per bet.
    </div>

    <!-- Table -->
    <div style="padding:20px 28px;overflow-x:auto;">
      <table style="width:100%;border-collapse:collapse;font-size:14px;">
        <thead>
          <tr style="background:#f0f0f0;text-align:left;">
            <th style="padding:10px 8px;font-size:12px;color:#666;text-transform:uppercase;">Rank</th>
            <th style="padding:10px 8px;font-size:12px;color:#666;text-transform:uppercase;">Match</th>
            <th style="padding:10px 8px;font-size:12px;color:#666;text-transform:uppercase;">Bet</th>
            <th style="padding:10px 8px;font-size:12px;color:#666;text-transform:uppercase;text-align:center;">Model %</th>
            <th style="padding:10px 8px;font-size:12px;color:#666;text-transform:uppercase;text-align:center;">Kalshi Price</th>
            <th style="padding:10px 8px;font-size:12px;color:#666;text-transform:uppercase;text-align:center;">Edge</th>
            <th style="padding:10px 8px;font-size:12px;color:#666;text-transform:uppercase;text-align:center;">Kelly</th>
            <th style="padding:10px 8px;font-size:12px;color:#666;text-transform:uppercase;text-align:center;">Volume</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>

    <!-- Footer -->
    <div style="padding:16px 28px;background:#f8f9fa;border-top:1px solid #eee;
                font-size:12px;color:#999;">
      These are paper trade predictions only. Always do your own research before placing real bets.
      Picks are sorted: upset picks first (highest edge), then value picks (highest edge).
    </div>
  </div>
</body>
</html>"""


def send_picks_email(bets: list[tuple]):
    """Send the daily picks email. bets = list of (match_dict, bet_dict)."""
    if not GMAIL_ADDRESS or not GMAIL_PASSWORD:
        print("[Email] Missing GMAIL_ADDRESS or GMAIL_APP_PASSWORD in .env — skipping email.")
        return

    today   = datetime.now(timezone.utc).strftime("%b %d, %Y")
    subject = f"Polybot ATP Picks · {today} · {len(bets)} bet(s)"
    html    = _build_html(bets)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = TO_ADDRESS
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, TO_ADDRESS, msg.as_string())

    print(f"[Email] Sent '{subject}' → {TO_ADDRESS}")
