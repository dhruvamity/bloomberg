# usage: python3 build_report.py analysis/report_data.json gold_news_shock_map.html
import sys, os
here = os.path.dirname(os.path.abspath(__file__))
t = open(os.path.join(here, "template.html")).read()
d = open(sys.argv[1]).read().replace("</", "<\\/")
open(sys.argv[2], "w").write(t.replace("__DATA__", d))
