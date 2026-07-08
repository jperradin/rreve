import os
from typing import List

from ...config.settings import Settings
from ...core.frame import Frame
from ...core.node import Node
from ...io.writer.base_writer import BaseWriter


class XYZDecorator(BaseWriter):
    """Append-mode extended-XYZ writer that decorates each frame with per-atom
    analysis outputs: coordination, polyhedricity, and particle index."""

    PROPERTIES = (
        "species:S:1:"
        "particle_index:I:1:"
        "pos:R:3:"
        "coordination:I:1:"
        "polyhedricity:R:1"
    )

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._settings: Settings = settings

        self._source_file = os.path.basename(self._settings.file_location)
        self._export_directory = os.path.join(
            self._settings.export_directory, "decorated_input_files"
        )
        self._destination_file = os.path.join(
            self._export_directory, self._source_file
        )
        self._mode = "a"

        os.makedirs(self._export_directory, exist_ok=True)

    def write(self, frame: Frame) -> None:
        num_nodes = frame.get_num_nodes()
        lattice_str = frame._lattice_str
        nodes = frame.get_nodes()

        with open(self._destination_file, self._mode) as f:
            f.write(self._header(num_nodes, lattice_str))
            for node in nodes:
                f.write(self._format_row(node))

    def _header(self, num_nodes: int, lattice_str: str) -> str:
        return (
            f"{num_nodes}\n"
            f'Lattice="{lattice_str}" Properties={self.PROPERTIES}\n'
        )

    @staticmethod
    def _format_row(node: Node) -> str:
        x, y, z = node.position
        return (
            f"{node.symbol}\t"
            f"{node.node_id}\t"
            f"{x:.6f}\t{y:.6f}\t{z:.6f}\t"
            f"{node.coordination}\t"
            f"{node.polyhedricity}\n"
        )
