import json
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


TEMPLATE_PATH = "Template.docx"
EXPORT_PATH = "template_layout.json"


def iter_block_items(parent):
    parent_elm = parent.element.body

    for child in parent_elm.iterchildren():

        if child.tag.endswith("}p"):
            yield Paragraph(child, parent)

        elif child.tag.endswith("}tbl"):
            yield Table(child, parent)


def extract_images(document):
    images = []
    rels = document.part.rels
    image_index = 1

    for rel in rels.values():

        if "image" in rel.target_ref:

            images.append({
                "id": f"image{image_index}",
                "target": rel.target_ref,
                "placeholder": f"image{image_index}",
                "type": "image"
            })

            image_index += 1

    return images


def table_to_structure(table, index):
    rows = []

    for row in table.rows:
        row_data = []

        for cell in row.cells:
            row_data.append(cell.text.strip())

        rows.append(row_data)

    headers = rows[0] if rows else []
    data = rows[1:] if len(rows) > 1 else []

    return {
        "id": f"table{index}",
        "type": "table",
        "header_variable": f"header_t{index}",
        "data_variable": f"data_t{index}",
        "headers": headers,
        "rows": data
    }


def paragraph_to_structure(text, index):
    return {
        "id": f"paragraph{index}",
        "type": "paragraph",
        "variable": f"paragraph{index}",
        "text": text
    }


def heading_to_structure(text):
    return {
        "id": "heading",
        "type": "heading",
        "variable": "heading",
        "text": text
    }


def export_layout(template_path):

    doc = Document(template_path)

    export_data = {
        "template": template_path,
        "layout": {
            "headings": [],
            "paragraphs": [],
            "tables": [],
            "images": []
        },
        "context_structure": {}
    }

    paragraph_index = 1
    table_index = 1

    for block in iter_block_items(doc):

        if isinstance(block, Paragraph):

            text = block.text.strip()

            if not text:
                continue

            style_name = block.style.name.lower()

            if "heading" in style_name:

                heading_data = heading_to_structure(text)

                export_data["layout"]["headings"].append(heading_data)

                export_data["context_structure"]["heading"] = text

            else:

                para_data = paragraph_to_structure(
                    text,
                    paragraph_index
                )

                export_data["layout"]["paragraphs"].append(para_data)

                export_data["context_structure"][
                    f"paragraph{paragraph_index}"
                ] = text

                paragraph_index += 1

        elif isinstance(block, Table):

            table_data = table_to_structure(
                block,
                table_index
            )

            export_data["layout"]["tables"].append(table_data)

            export_data["context_structure"][
                f"header_t{table_index}"
            ] = table_data["headers"]

            export_data["context_structure"][
                f"data_t{table_index}"
            ] = table_data["rows"]

            table_index += 1

    images = extract_images(doc)

    export_data["layout"]["images"] = images

    for image in images:

        export_data["context_structure"][
            image["placeholder"]
        ] = image["target"]

    return export_data


if __name__ == "__main__":

    result = export_layout(TEMPLATE_PATH)

    with open(EXPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"[+] Layout exported -> {EXPORT_PATH}")

    print(json.dumps(
        result["context_structure"],
        indent=4,
        ensure_ascii=False
    ))