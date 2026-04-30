from ...config.settings import Settings
from ...io.writer.base_writer import BaseWriter
from ...core.frame import Frame

import os

class XYZDecorator(BaseWriter):
    def __init__(self, settings:Settings) -> None:
        super().__init__(settings)
        self._settings: Settings = settings

        self._source_file = self._settings.file_location.split('/')[-1]
        self._export_directory = os.path.join(self._settings.export_directory, 'decorated_input_files') 
        self._destination_file = os.path.join(self._export_directory, self._source_file)
        self._mode = 'a'

    def write(self, f:Frame) -> None:

        _num_nodes = f.get_num_nodes()
        _lattice = f._lattice_str
        _nodes = f.get_nodes()

        # properties output
        # Symbol X Y Z Coordination Polyhedricity
        with open(self._destination_file, self._mode) as o:
            o.write(f"{_num_nodes}\n")
            o.write(f"Lattice=\"{_lattice}\" Properties=species:S:1:pos:R:3:coordination:I:1:polyhedricity:R:1\n")
            for n in _nodes:
                o.write(f"{n.symbol}\t{n.position[0]}\t{n.position[1]}\t{n.position[2]}\t{n.coordination}\t{n.polyhedricity}\n")
