"""
Scan Kalshi for value bets on open ATP matches.

Usage:
  python scan.py              # scan + log paper picks + send email
  python scan.py --no-email   # scan + log, skip email
  python scan.py --no-log     # scan without logging or email
  python scan.py --report     # show paper trading performance
"""
import sys
sys.path.insert(0, "src")

if "--report" in sys.argv:
    from tracker import report
    report()
else:
    from scanner import scan
    from emailer import send_picks_email

    log   = "--no-log"   not in sys.argv
    email = "--no-email" not in sys.argv and "--no-log" not in sys.argv

    bets = scan(log_picks=log)

    if email and bets:
        send_picks_email(bets)
    elif email and not bets:
        print("[Email] No picks today — skipping email.")
