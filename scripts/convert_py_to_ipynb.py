"""Convertit un .py avec # === CELL N === markers en .ipynb."""
import nbformat as nbf
import sys

def convert_py_to_ipynb(py_path: str, ipynb_path: str):
    with open(py_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cells = []
    current_lines = []
    in_cell = False

    for line in lines:
        # Détecter un nouveau marqueur de cellule: # === CELL N ===
        if line.startswith("# === CELL ") and line.strip().endswith("==="):
            if in_cell and current_lines:
                # Sauvegarder la cellule précédente
                code = "".join(current_lines).strip()
                if code:
                    cells.append(code)
                current_lines = []
            in_cell = True
            continue
        
        if in_cell:
            current_lines.append(line)

    # Dernière cellule
    if current_lines:
        code = "".join(current_lines).strip()
        if code:
            cells.append(code)

    # Construire le notebook
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    }

    for code in cells:
        nb.cells.append(nbf.v4.new_code_cell(code))

    with open(ipynb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Converted {py_path} → {ipynb_path} ({len(cells)} cells)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_py_to_ipynb.py <input.py> <output.ipynb>")
        sys.exit(1)
    convert_py_to_ipynb(sys.argv[1], sys.argv[2])
