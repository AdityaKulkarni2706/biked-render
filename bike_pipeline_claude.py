"""
bike_pipeline.py
----------------
Pipeline for converting BIKED dataset rows into SVG visualisations.
Injects a row's features directly into a BikeCAD XML template (.bcad),
bypassing processGen entirely.

Typical usage:
    from bike_pipeline import BikePipeline

    with BikePipeline() as pipeline:
        svg_path = pipeline.row_to_svg(df.iloc[5])

Or without context manager:
    pipeline = BikePipeline()
    svg_path = pipeline.row_to_svg(df.iloc[5])
    pipeline.close()
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from GA_Clip_utils import BikeCAD


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TEMPLATE_PATH = r"Biked_Reference_Data\PlainRoadbikestandardized.txt"
DEFAULT_BCAD_DIR      = r"Biked_Reference_Data\output\bcad"
DEFAULT_SVG_DIR       = r"Biked_Reference_Data\output\svg"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class BikePipeline:
    """
    Converts a BIKED DataFrame row → .bcad → .svg via direct XML injection.

    Parameters
    ----------
    template_path : str
        Path to the base BikeCAD XML template.
    bcad_dir : str
        Directory where .bcad files are saved.
    svg_dir : str
        Directory where .svg files are saved.
    """
    DEFAULT_TEMPLATE_PATH = r"Biked_Reference_Data\PlainRoadbikestandardized.txt"
    DEFAULT_BCAD_DIR      = r"Biked_Reference_Data\output\bcad"
    DEFAULT_SVG_DIR       = r"Biked_Reference_Data\output\svg"
    def __init__(
        self,
        template_path: str = DEFAULT_TEMPLATE_PATH,
        bcad_dir:      str = DEFAULT_BCAD_DIR,
        svg_dir:       str = DEFAULT_SVG_DIR,
    ):
        self.template_path = Path(template_path)
        self.bcad_dir      = Path(bcad_dir)
        self.svg_dir       = Path(svg_dir)

        self.bcad_dir.mkdir(parents=True, exist_ok=True)
        self.svg_dir.mkdir(parents=True, exist_ok=True)

        self._cad_engine = BikeCAD()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def row_to_svg(self, row: pd.DataFrame | pd.Series, name: str = None) -> Path:
        """
        Full pipeline: DataFrame row → .bcad → .svg.

        Parameters
        ----------
        row : pd.DataFrame | pd.Series
            A single row, either as a Series (df.iloc[5]) or a
            single-row DataFrame (df.iloc[[5]]).
        name : str, optional
            Stem for the output filenames (e.g. "bike_005").
            Defaults to the row's index label.

        Returns
        -------
        Path
            Path to the generated .svg file.
        """
        series    = self._coerce_to_series(row)
        file_stem = name or self._infer_name(series)

        bcad_path = self._series_to_bcad(series, file_stem)
        svg_path  = self._bcad_to_svg(bcad_path)
        return svg_path

    def close(self):
        """Shut down the BikeCAD engine. Call when done."""
        self._cad_engine.kill()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _coerce_to_series(self, row: pd.DataFrame | pd.Series) -> pd.Series:
        """Normalise input to a pd.Series regardless of whether a
        single-row DataFrame or Series was passed in."""
        if isinstance(row, pd.DataFrame):
            return row.iloc[0]
        return row

    def _infer_name(self, series: pd.Series) -> str:
        """Derive a filename stem from the series index label."""
        return str(series.name or "bike")

    def _series_to_bcad(self, series: pd.Series, file_stem: str) -> Path:
        """
        Inject a row's features into the XML template and write a .bcad file.

        Only keys present in both the template and the Series are written.
        NaN values are skipped. mmInch entries are forced to "1" (mm mode).
        Java-style booleans and integer-valued floats are formatted correctly.
        """
        row_dict = series.to_dict()

        tree = ET.parse(self.template_path)
        root = tree.getroot()

        updated_count = 0

        for entry in root.findall("entry"):
            key = entry.get("key")

            # Force millimetre units
            if key and key.endswith("mmInch"):
                entry.text = "1"
                continue

            if key not in row_dict:
                continue

            val = row_dict[key]

            if pd.isna(val):
                continue

            entry.text     = self._format_value(val)
            updated_count += 1

        bcad_path = self.bcad_dir / f"{file_stem}.bcad"
        self._write_bcad(root, bcad_path)

        print(f"[bcad] Injected {updated_count} features → {bcad_path.name}")
        return bcad_path

    def _bcad_to_svg(self, bcad_path: Path) -> Path:
        """Export a .bcad file to .svg via the BikeCAD engine."""
        self._cad_engine.export_svg_from_list([str(bcad_path)])

        # THE FIX: Look for the SVG in the same directory as the .bcad file, 
        # not the intended svg_dir.
        actual_svg_path = bcad_path.with_suffix(".svg")
        
        # Optional but highly recommended: Verify the engine actually made it!
        if not actual_svg_path.exists():
            print(f"[ERROR] BikeCAD failed to generate SVG at {actual_svg_path}")
            
        print(f"[svg]  Rendered -> {actual_svg_path.name} (in bcad folder)")
        return actual_svg_path

    @staticmethod
    def _format_value(val) -> str:
        """
        Format a Python/NumPy value into a BikeCAD-compatible string.
        - Booleans   → lowercase "true" / "false"  (Java expects this)
        - Float n.0  → "n"                          (BikeCAD rejects "34.0")
        - Everything else → str()
        """
        if isinstance(val, (bool, np.bool_)):
            return "true" if val else "false"
        if isinstance(val, (float, np.floating)) and val.is_integer():
            return str(int(val))
        return str(val)

    @staticmethod
    def _write_bcad(root: ET.Element, path: Path) -> None:
        """Write an ElementTree root to a .bcad file with the Java XML header."""
        java_header = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE properties SYSTEM "http://java.sun.com/dtd/properties.dtd">\n'
        )
        xml_body = ET.tostring(root, encoding="utf-8", xml_declaration=False).decode("utf-8")
        path.write_text(java_header + xml_body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Smoke test  (python bike_pipeline.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv(r"Biked_Reference_Data\clip_sBIKED_processed.csv")

    with BikePipeline() as pipeline:
        for i in [0, 1, 5, 10]:
            svg = pipeline.row_to_svg(df.iloc[i], f"bike_{i}")
            print(f"  → {svg}\n")
