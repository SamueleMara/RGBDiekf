"""Production imports enforce two independent estimators and shared data code."""
import ast,unittest
from pathlib import Path


class ArchitectureTests(unittest.TestCase):
    def test_production_estimators_are_independent_and_common_does_not_select_one(self):
        root=Path(__file__).resolve().parents[1]
        modules=[*root.glob('main*.py'),*(root/'rgbd_common').rglob('*.py'),*(root/'iekf').rglob('*.py'),*(root/'ukf').rglob('*.py')]
        for path in modules:
            relative=path.relative_to(root);first=relative.parts[0]
            forbidden={'rgbd','tests','oracle'}
            if first in ('iekf','main_iekf.py','main.py'):forbidden.add('ukf')
            if first in ('ukf','main_ukf.py'):forbidden.add('iekf')
            if first=='rgbd_common':forbidden.update(('iekf','ukf'))
            for node in ast.walk(ast.parse(path.read_text())):
                names=[node.module] if isinstance(node,ast.ImportFrom) and node.level==0 else ([a.name for a in node.names] if isinstance(node,ast.Import) else [])
                for name in names:
                    self.assertNotIn(name.split('.')[0],forbidden,f'{relative}: prohibited production import {name}')
        self.assertFalse((root/'rgbd').exists())
