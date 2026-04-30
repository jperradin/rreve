from typing import Optional

from ..config.settings import Settings
from .analyzers.base_analyzer import BaseAnalyzer
from .analyzers.neutron_structure_factor_analyzer import NeutronStructureFactorAnalyzer
from .analyzers.neutron_structure_factor_fft_analyzer import (
    NeutronStructureFactorFFTAnalyzer,
)
from .analyzers.pair_distribution_function_analyzer import (
    PairDistributionFunctionAnalyzer,
)
from .analyzers.bond_angular_distribution_analyzer import (
    BondAngularDistributionAnalyzer,
)
from .analyzers.structural_units_analyzer import StructuralUnitsAnalyzer
from .analyzers.connectivity_analyzer import ConnectivityAnalyzer
from .analyzers.polyhedricity_analyzer import PolyhedricityAnalyzer


class AnalyzerFactory:
    def __init__(self, settings: Settings, verbose: bool = True):
        self._analyzers = {}
        # Register other analyzers here
        self.register_analyzer(NeutronStructureFactorAnalyzer(settings))
        self.register_analyzer(NeutronStructureFactorFFTAnalyzer(settings))
        self.register_analyzer(PairDistributionFunctionAnalyzer(settings))
        self.register_analyzer(BondAngularDistributionAnalyzer(settings))
        self.register_analyzer(StructuralUnitsAnalyzer(settings))
        self.register_analyzer(ConnectivityAnalyzer(settings))
        self.register_analyzer(PolyhedricityAnalyzer(settings))

    def register_analyzer(self, analyzer: BaseAnalyzer) -> None:
        self._analyzers[analyzer.__class__.__name__] = analyzer

    def get_analyzer(self, analyzer_name: str) -> Optional[BaseAnalyzer]:
        return self._analyzers.get(analyzer_name)
