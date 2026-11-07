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

_EAGER = frozenset({"sources", "objects", "outputs"})

@dataclass
class ResumeData:
    entity_manager: Any

    def __post_init__(self):
        config = self.entity_manager.get("config")
        self.status_dir = f"{config.system.cache_path}/status/{__class__.__name__}"
        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)

    def load_data(self, eager_only: bool = False):
        store = self.entity_manager._store
        for name in store.DEFAULT_ENTITIES:
            if eager_only and name not in _EAGER:
                continue
            keys = store.keys(name)
            if not keys:
                continue

            # eager path already done by EntityManager._hydrate_eager
            if name in _EAGER:
                continue

            # Lazy collections: DON'T hydrate now. Just make sure they're
            # discoverable later via EntityManager.get(name) — which pulls
            # from the store on demand.
            debug_print(f"[ResumeData] {name}: {len(keys)} items available lazily")
