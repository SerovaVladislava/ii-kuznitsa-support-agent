"""Коммерческие предложения и счета: создание, нумерация, вывод в HTML (печать/PDF) и DOCX."""
from __future__ import annotations

import io
import secrets
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import db
from ..config import get_settings
from . import settings_store
from .textutil import amount_in_words, fmt_date, fmt_money, fmt_qty

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "web" / "templates"

KIND_TITLES = {"quote": "Коммерческое предложение", "invoice": "Счёт на оплату"}

_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=select_autoescape(["html"]))
_env.filters["money"] = fmt_money
_env.filters["qty"] = fmt_qty


def _next_number(kind: str) -> str:
    year = date.today().year
    count = db.scalar("SELECT COUNT(*) FROM documents WHERE kind = ? AND substr(created_at, 1, 4) = ?",
                      (kind, str(year))) or 0
    seq = count + 1
    return f"КП-{year}-{seq:04d}" if kind == "quote" else f"{year % 100:02d}-{seq:04d}"


def public_url(doc: dict[str, Any]) -> str:
    return f"{get_settings().public_base_url}/d/{doc['token']}"


def _create(kind: str, calc: dict[str, Any], buyer: dict[str, Any], dialog_id: int | None,
            customer_id: int | None, quote_id: int | None = None) -> dict[str, Any]:
    if not calc.get("lines"):
        raise ValueError("Нет позиций для документа")
    rules = settings_store.get("pricing")
    days = rules.get("quote_valid_days" if kind == "quote" else "invoice_valid_days", 5)
    valid_until = (date.today() + timedelta(days=int(days))).isoformat()
    totals = {k: v for k, v in calc.items() if k != "lines"}
    doc_id = db.execute(
        "INSERT INTO documents (kind, number, token, dialog_id, customer_id, quote_id, buyer, items, totals, "
        "status, valid_until, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?, ?)",
        (kind, _next_number(kind), secrets.token_urlsafe(16), dialog_id, customer_id, quote_id,
         db.dumps(buyer), db.dumps(calc["lines"]), db.dumps(totals), valid_until, db.now_iso()),
    )
    doc = get(doc_id)
    db.log_event(f"{kind}_created", dialog_id, document_id=doc_id, number=doc["number"], total=calc["total"])
    return doc


def create_quote(calc: dict[str, Any], buyer: dict[str, Any], dialog_id: int | None = None,
                 customer_id: int | None = None) -> dict[str, Any]:
    return _create("quote", calc, buyer, dialog_id, customer_id)


def create_invoice(calc: dict[str, Any], buyer: dict[str, Any], dialog_id: int | None = None,
                   customer_id: int | None = None, quote_id: int | None = None) -> dict[str, Any]:
    if not (buyer.get("company") or buyer.get("name")):
        raise ValueError("Для счёта нужен покупатель: организация (и ИНН) или ФИО")
    return _create("invoice", calc, buyer, dialog_id, customer_id, quote_id)


def _hydrate(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    row["buyer"] = db.loads(row["buyer"], {})
    row["items"] = db.loads(row["items"], [])
    row["totals"] = db.loads(row["totals"], {})
    row["title"] = KIND_TITLES.get(row["kind"], row["kind"])
    row["url"] = public_url(row)
    return row


def get(doc_id: int) -> dict[str, Any] | None:
    return _hydrate(db.query_one("SELECT * FROM documents WHERE id = ?", (doc_id,)))


def get_by_token(token: str) -> dict[str, Any] | None:
    return _hydrate(db.query_one("SELECT * FROM documents WHERE token = ?", (token,)))


def get_by_number(number: str) -> dict[str, Any] | None:
    return _hydrate(db.query_one("SELECT * FROM documents WHERE number = ? ORDER BY id DESC LIMIT 1",
                                 (number.strip(),)))


def list_documents(dialog_id: int | None = None, limit: int = 200) -> list[dict[str, Any]]:
    if dialog_id:
        rows = db.query("SELECT * FROM documents WHERE dialog_id = ? ORDER BY id DESC", (dialog_id,))
    else:
        rows = db.query("SELECT * FROM documents ORDER BY id DESC LIMIT ?", (limit,))
    return [_hydrate(r) for r in rows]


STATUS_LABELS = {"draft": "черновик, на проверке", "approved": "проверен менеджером", "sent": "отправлен",
                 "paid": "оплачен", "cancelled": "отменён"}


def set_status(doc_id: int, status: str) -> None:
    db.execute("UPDATE documents SET status = ? WHERE id = ?", (status, doc_id))
    if status == "paid":
        doc = get(doc_id)
        db.log_event("invoice_paid", doc["dialog_id"], document_id=doc_id, total=doc["totals"].get("total"))


def buyer_line(buyer: dict[str, Any]) -> str:
    parts = []
    if buyer.get("company"):
        parts.append(buyer["company"])
    if buyer.get("inn"):
        parts.append(f"ИНН {buyer['inn']}")
    if buyer.get("kpp"):
        parts.append(f"КПП {buyer['kpp']}")
    if buyer.get("address"):
        parts.append(buyer["address"])
    if not buyer.get("company") and buyer.get("name"):
        parts.append(buyer["name"])
    return ", ".join(parts)


def _context(doc: dict[str, Any]) -> dict[str, Any]:
    created = datetime.fromisoformat(doc["created_at"]).date()
    return {
        "doc": doc,
        "company": settings_store.get("company"),
        "created": fmt_date(created),
        "valid_until": fmt_date(date.fromisoformat(doc["valid_until"])) if doc["valid_until"] else "",
        "buyer_line": buyer_line(doc["buyer"]),
        "buyer_name": doc["buyer"].get("company") or doc["buyer"].get("name") or "",
        "payment_terms": (doc["totals"].get("payment_terms") or "100% предоплата").replace("100% предоплата",
                                                                                         "Предоплата 100%"),
        "status_label": STATUS_LABELS.get(doc["status"], doc["status"]),
        "amount_words": amount_in_words(doc["totals"]["total"]),
    }


def render_html(doc: dict[str, Any]) -> str:
    template = _env.get_template(f"docs/{doc['kind']}.html")
    return template.render(**_context(doc))


def render_docx(doc: dict[str, Any]) -> bytes:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, RGBColor

    ctx = _context(doc)
    company, totals = ctx["company"], doc["totals"]
    d = Document()
    style = d.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10)

    if doc["status"] == "draft":
        p = d.add_paragraph()
        r = p.add_run("ЧЕРНОВИК — на проверке у менеджера")
        r.bold = True
        r.font.color.rgb = RGBColor(0xC0, 0x1E, 0x2E)

    head = d.add_table(rows=1, cols=2)
    left, right = head.rows[0].cells
    left.text = "\n".join([company["name"], company["address"], f"Т {company['phone']} | {company['email']}",
                           company["website"]])
    if doc["kind"] == "quote":
        right.text = f"Куда: {ctx['buyer_name'] or '—'}\nИсх. № {doc['number']} от {ctx['created']}."
    else:
        right.text = f"Счёт на оплату № {doc['number']} от {ctx['created']}"
    right.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT

    if doc["kind"] == "quote":
        p = d.add_paragraph(company.get("quote_salutation") or "Уважаемый заказчик!")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        d.add_paragraph(f"{company['name']} предлагает Вашему вниманию коммерческое предложение на следующих "
                        f"условиях:").alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        d.add_paragraph(f"Поставщик: {company['name']}" + (f", ИНН {company['inn']}" if company.get("inn") else "")
                        + f", {company['address']}")
        d.add_paragraph(f"Покупатель: {ctx['buyer_line']}")

    cols = ["Артикул", "Наименование", "Кол-во", "Цена, руб.", "Сумма, руб.", "В т.ч. НДС, руб."]
    table = d.add_table(rows=1, cols=len(cols))
    table.style = "Table Grid"
    for i, c in enumerate(cols):
        table.rows[0].cells[i].text = c
        table.rows[0].cells[i].paragraphs[0].runs[0].bold = True
    for ln in doc["items"]:
        cells = table.add_row().cells
        vat = fmt_money(ln.get("vat_amount", 0)) if ln.get("vat_rate") else "Без НДС"
        for i, v in enumerate([ln["sku"], ln["name"], fmt_qty(ln["quantity"]), fmt_money(ln["price_final"]),
                               fmt_money(ln["amount"]), vat]):
            cells[i].text = v
    cells = table.add_row().cells
    cells[1].text = "Итого"
    cells[4].text = fmt_money(totals["total"])
    cells[5].text = fmt_money(totals["vat_amount"]) if totals.get("vat_amount") else "Без НДС"
    for c in (cells[1], cells[4], cells[5]):
        c.paragraphs[0].runs[0].bold = True

    d.add_paragraph()
    notes = [company.get("quote_price_note") or "Цена указана с учетом всех налогов, сборов, и других "
                                                  "обязательных платежей."]
    if totals.get("discount_percent"):
        notes.append(f"Цены указаны с учётом скидки {totals['discount_percent']:g}%.")
    notes.append(f"Условия оплаты: {ctx['payment_terms']}.")
    notes.append(f"Срок поставки: {totals.get('delivery_terms') or 'по согласованию'}.")
    notes.append(f"Срок действия {'коммерческого предложения' if doc['kind'] == 'quote' else 'счёта'} - до "
                 f"{ctx['valid_until']}.")
    notes.append(f"{ctx['amount_words']}.")
    for n in notes:
        d.add_paragraph(n)
    d.add_paragraph()
    d.add_paragraph(f"{company.get('signer_name') or '_______________'}\n"
                    f"{company.get('signer_title') or 'Ответственный менеджер'}")

    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()
