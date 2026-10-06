# Copyright (C) 2025 Malcom3D <malcom3d.gpl@gmail.com>
#
# This file is part of pbrAudio.
#
# pbrAudio is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pbrAudio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pbrAudio.  If not, see <https://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPL-3.0-or-later

import copy
#import threading
import numpy as np
from typing import List, Tuple, Any
from ..utils.config import Config
#from ..lib.functions import _soxel_grid_shape

class EntityManager:
    _instance = None
#    _lock = threading.Lock()
    _initialized = False
    
#    def __new__(cls, *args, **kwargs):
#        with cls._lock:
#            if cls._instance is None:
#                cls._instance = super().__new__(cls)
#        return cls._instance
    
    def __init__(self, config: str):
#        with self._lock:
        if not self._initialized:
            self._sources = {}
            self._objects = {}
            self._outputs = {}
            self._output_datas = {}
            self._wave_propagators = {}
            self._layer_managers = {}
            self._trajectories = {}
            self._collisions = {}
            self._forces = {}
            self._modal_vertices = {}
            self._score_tracks = {}
            self._fracture_events = {}
            self._rigidbody_synth = {}
            self._resonance_synth = {}
            self._singleton = {}
            self._initialized = True

            self.sigleton_map = {
                'config': 'Config',
                'frames': 'FrameCounter',
                'sample_counter': 'SampleCounter',
                'connected_buffer': 'ConnectedBuffer',
                'frequency_bands': 'FrequencyBands',
                'soxel_grid': 'SoxelGrid',
                'geometry_data': 'GeometryData',
                'material_properties': 'MaterialProperties',
                'medium_properties': 'MediumProperties'
            }
            self.entities_map = {
                'sources': ['ParticlesSource', 'ObjectSource', 'EnvironmetSource', 'SphericalSource', 'PlanarSource'],
                'objects': ['AcousticObject', 'SurfaceVoxelObject'],
                'outputs': ['AmbisonicOutput', 'OmnidirectionalOutput', 'Figure8Output', 'CardioidOutput', 'HypercardioidOutput'],
                'wave_propagators': 'WavePropagator',
                'output_datas': 'OutputData',
                'trajectories': ['ParticlesTrajectoryData', 'TrajectoryData', 'tmpTrajectoryData'],
                'collisions': [ 'CollisionData', 'ParticlesCollisionsPoints', 'ParticlesCollisionsVoxels'],
                'forces': [ 'ForceData', 'ForceDataSequence'],
                'modal_vertices': 'ModalVertices',
                'score_tracks': 'ScoreTrack',
                'fracture_events': 'FractureEvent',
                'rigidbody_synth': 'RigidBodySynth',
                'resonance_synth': 'ResonanceSynth'
            }

            config = Config(config)
            self.register('config', config)

    # Dispatcher:
    def register(self, entity: str, obj: Any) -> int:
        if entity in self.sigleton_map and not entity in self._singleton:
            self._singleton[entity] = obj
#        elif entity in self.entities_map.keys() and not idx == None: # when entity is unregistered len(List(idx]) != List[idx]
        elif entity in self.entities_map.keys():
            for key in self.entities_map.keys():
                if entity in key:
                    for sub in self.entities_map[key]:
                        if sub in str(type(obj)):
                            entities = eval(f"self._{key}")
                            idx = 0
                            if not len(entities.keys()) == 0:
                                idx = list(entities.keys())[-1] + 1
                            entities[idx] = obj
                            return idx
               
    def get(self, entity: str = None, idx: int = None) -> dict[str, Any]:
        """Get all objects"""
        if entity == None:
            return self._singleton, self._sources, self._objects, self._outputs, self._wave_propagators, self._output_datas, self._trajectories, self._collisions, self._forces, self._modal_vertices, self._score_tracks, self._fracture_events, self._rigidbody_synth, self._resonance_synth
        for key in self.sigleton_map.keys():
            if entity in key:
                if entity in ['geometry_data', 'material_properties', 'medium_properties']:
                    return copy.deepcopy(self._singleton[entity])
                return self._singleton[entity]
            else:
                for key in self.entities_map.keys():
                    if entity in key:
                        entities = eval(f"self._{entity}")
                        return entities.get(idx) if not idx == None else entities

    def count_entity(self, entity: str = None):
        """Count all objects"""
        entities_count = {}
        if entity == None:
            for entity in self.sigleton_map.keys():
                if entity in self._singleton:
                    entities_count[entity] = 1
                elif entity not in self._singleton:
                    entities_count[entity] = 0
            for entity in self.entities_map.keys():
                _entity = eval(f"self._{entity}")
                entities_count[entity] = len(_entity)

        elif entity in self.sigleton_map.keys():
            if entity in self._singleton:
                entities_count[entity] = 1
            else:
                entities_count[entity] = 0

        elif entity in self.entities_map.keys():
            _entity = eval(f"self._{entity}")
            entities_count[entity] = len(_entity)
        
        return entities_count

    def unregister(self, entity: str, idx: int = None) -> None:
        """Unregister an object"""
        for key in self.sigleton_map.keys():
            if entity in key:
                del self._singleton[entity]
            elif not idx == None:
                for key in self.entities_map.keys():
                    if entity in key:
                        entities = eval(f"self._{entity}")
                        if idx in entities.keys():
                            del entities[idx]
                            return entities

    def dump(self, is_score_track_final: bool = False):
        # Ensure directory exists
        trajectories_dir = f"{config.system.cache_path}/trajectories"
        collisions_dir = f"{self._config.system.cache_path}/collisions"
        forces_dir = f"{self._config.system.cache_path}/forces_data"
        modalvertices_dir = f"{self._config.system.cache_path}/modalvertices"
        scoretracks_dir = f"{self._config.system.cache_path}/scoretracks"

        os.makedirs(trajectories_dir, exist_ok=True)
        os.makedirs(collisions_dir, exist_ok=True)
        os.makedirs(modalvertices_dir, exist_ok=True)
        os.makedirs(scoretracks_dir, exist_ok=True)
        os.makedirs(forces_dir, exist_ok=True)

        # Save forces data
        for force_idx in self._forces.keys():
            if isinstance(self._forces[force_idx], ForceDataSequence):
                force_obj_idx = self._forces[force_idx].obj_idx
                force_other_obj_idx = self._forces[force_idx].other_obj_idx
                self._forces[force_idx].save(f"{self.forces_dir}/{force_obj_idx:05d}_{force_other_obj_idx:05d}.pkl")
        print('Saved force data: ', len(self._forces))

        # Save collision data
        for c_idx in self._collisions.keys():
            self._collisions[c_idx].save(f"{self.collisions_dir}/{c_idx:05d}.pkl")
        print('Saved collisions: ', len(self._collisions))

        # Save modal vertices data
        for m_idx in self._modal_vertices.keys():
            self._modal_vertices[m_idx].save(f"{self.modalvertices_dir}/{m_idx:05d}.json")
        print('Saved modal_vertices: ', len(self._modal_vertices))

        if not is_score_track_final:
            # Save score tracks data
            for s_idx in self._score_tracks.keys():
                self._score_tracks[s_idx].save(f"{self.scoretracks_dir}/{s_idx:05d}.tar.gz")
            print('Saved score_tracks: ', len(self._score_tracks))
        elif is_score_track_final:
            # Save score tracks data in /tmp
            n_score = []
            for s_idx in self._score_tracks.keys():
                if self._score_tracks[s_idx].is_final:
                    self._score_tracks[s_idx].save(f"/tmp/{s_idx:05d}.tar.gz")
                    n_score += [f"/tmp/{s_idx:05d}.tar.gz"]

            # Clean score tracks data
            if os.path.exists(self.scoretracks_dir):
                filenames = os.listdir(self.scoretracks_dir)
                for filename in filenames:
                    if os.path.isfile(f"{self.scoretracks_dir}/{filename}"):
                        os.remove(f"{self.scoretracks_dir}/{filename}")

            # Move score tracks data files from /tmp
            for filename in n_score:
                shutil.move(filename, f"{self.scoretracks_dir}/{filename.removeprefix('/tmp/')}")
            print('Saved final score_tracks: ', len(n_score))
