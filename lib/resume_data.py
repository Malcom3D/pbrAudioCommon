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

import os
import numpy as np
from typing import List, Tuple, Any, Dict
from dataclasses import dataclass, field
from dask import delayed, compute

from pbrAudioCommon import EntityManager, CollisionData, ForceDataSequence, ModalVertices, TrajectoryData, ScoreTrack

@dataclass
class ResumeData:
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.status_dir = f"{config.system.cache_path}/status/{__class__.__name__}"
        self.physical_core = config.system.physical_core
        self.collisions_dir = f"{config.system.cache_path}/collisions"
        self.trajectories_dir = f"{config.system.cache_path}/trajectories"
        self.forces_dir = f"{config.system.cache_path}/forces_data"
        self.modalvertices_dir = f"{config.system.cache_path}/modalvertices"
        self.scoretracks_dir = f"{config.system.cache_path}/scoretracks"

    def load_data(self):
        trajectories = self.entity_manager.get('trajectories')
        if len(trajectories) == 0:
            if os.path.exists(f"{self.trajectories_dir}") and not len(os.listdir(f"{self.trajectories_dir}")) == 0:
                for filename in os.listdir(f"{self.trajectories_dir}"):
                    trajectory = None
                    if filename.endswith('.pkl') and os.path.isfile(f"{self.trajectories_dir}/corrected/{filename}"):
                        trajectory = TrajectoryData.load(f"{self.trajectories_dir}/corrected/{filename}")
                    elif filename.endswith('.pkl') and not os.path.isfile(f"{self.trajectories_dir}/corrected/{filename}"):
                        trajectory = TrajectoryData.load(f"{self.trajectories_dir}/{filename}")
                    if trajectory is not None:
                        _ = self.entity_manager.register('trajectories', trajectory)

        collisions = self.entity_manager.get('collisions')
        if len(collisions) == 0:
            if os.path.exists(f"{self.collisions_dir}") and not len(os.listdir(f"{self.collisions_dir}")) == 0:
                for filename in os.listdir(f"{self.collisions_dir}"):
                    if filename.endswith('.pkl'):
                        collisions = CollisionData.load(f"{self.collisions_dir}/{filename}")
                        _ = self.entity_manager.register('collisions', collisions)

        forces = self.entity_manager.get('forces')
        if len(forces) == 0:
            if os.path.exists(f"{self.forces_dir}") and not len(os.listdir(f"{self.forces_dir}")) == 0:
                for filename in os.listdir(f"{self.forces_dir}"):
                    if filename.endswith('.pkl'):
                        forces = ForceDataSequence.load(f"{self.forces_dir}/{filename}")
                        _ = self.entity_manager.register('forces', forces)
            forces = self.entity_manager.get('forces')

        modal_vertices = self.entity_manager.get('modal_vertices')
        if len(modal_vertices) == 0:
            if os.path.exists(self.modalvertices_dir) and not len(os.listdir(f"{self.modalvertices_dir}")) == 0:
                filenames = os.listdir(self.modalvertices_dir)
                for filename in filenames:
                    if os.path.isfile(f"{self.modalvertices_dir}/{filename}"):
                        modal_vertices = ModalVertices.load(f"{self.modalvertices_dir}/{filename}")
                        _ = self.entity_manager.register('modal_vertices', modal_vertices)

        score_tracks = self.entity_manager.get('score_tracks')
        if len(score_tracks) == 0:
            if os.path.exists(self.scoretracks_dir) and not len(os.listdir(f"{self.scoretracks_dir}")) == 0:
                filenames = os.listdir(self.scoretracks_dir)
                for filename in filenames:
                    if os.path.isfile(f"{self.scoretracks_dir}/{filename}"):
                        score_tracks = ScoreTrack.load(f"{self.scoretracks_dir}/{filename}")
                        _ = self.entity_manager.register('score_tracks', score_tracks)
