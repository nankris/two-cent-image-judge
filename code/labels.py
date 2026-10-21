# RVL-CDIP label index -> option key + one-line description (same text for every arm)
LABELS = [
 ("advertisement", "An advertisement or promotional flyer"),
 ("budget", "A budget or financial table of planned/actual amounts"),
 ("email", "A printed email message"),
 ("file_folder", "The cover or tab of a file folder"),
 ("form", "A form with fields or boxes to fill in"),
 ("handwritten", "A mostly handwritten note or page"),
 ("invoice", "An invoice or bill requesting payment"),
 ("letter", "A formal letter with letterhead, salutation and signature"),
 ("memo", "An internal memorandum (To/From/Subject header)"),
 ("news_article", "A newspaper or magazine article"),
 ("presentation", "A presentation slide or overhead"),
 ("questionnaire", "A questionnaire or survey"),
 ("resume", "A resume or curriculum vitae"),
 ("publication", "A scientific journal publication or paper"),
 ("report", "A scientific or technical report"),
 ("specification", "A product or technical specification sheet"),
]
KEYS = [k for k, _ in LABELS]
CRITERIA = {k: d for k, d in LABELS}
STATE = "A scanned business document from a corporate archive."
QUESTION = {"name": "doctype", "instructions": "Which type of document is shown in the image?", "criteria": CRITERIA}
