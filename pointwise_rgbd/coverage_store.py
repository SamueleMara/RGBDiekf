"""Coverage is a many-to-one surface relation, never a measurement owner."""
import numpy as np
from rgbd_common.array_store import ArrayStore


class FrozenMeasurementConflict(RuntimeError):
    def __init__(self,reference,owner):
        self.reference=reference;self.owner=int(owner)
        super().__init__(f'Geometrically unexplained observation {reference} already belongs to frozen point {owner}; seeding it again would duplicate a physical measurement')


class CoverageIndex:
    def __init__(self,path):self.store=ArrayStore(path)
    def write(self,k,c,version,result):
        self.store.write(f'{k}:{c}',model_version=np.array(version),pixel=result.pixel,explained=result.explained,geometry_explained=result.geometry_explained,point_id=result.point_id,geometry_point_id=result.geometry_point_id)
    def read(self,k,c,version):
        record=self.store.read(f'{k}:{c}')
        # An unchanged block can be older than the completed global horizon.
        # A partially written future horizon must never be exposed to discovery.
        local=int(record['model_version'])
        if local>version or (local!=version and not self.completed(version)):
            raise ValueError('SURFACE_COVERAGE_STALE: finish coverage for this SOLVED version')
        return record
    def completed(self,version):
        return self.store.contains('_complete_version') and int(self.store.read('_complete_version')['version'])==version
    def horizon(self):
        return self.store.read('_complete_version') if self.store.contains('_complete_version') else None
    def complete(self,version,last_point_id=-1):
        self.store.write('_complete_version',version=np.array(version),last_point_id=np.array(last_point_id))
