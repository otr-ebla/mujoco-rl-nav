from __future__ import annotations
import os
import sys
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import gymnasium as gym
import mujoco
import mujoco.viewer


class mobilerobotRL(gym.Env):
    """
    Base class per ambienti MuJoCo del mobile robot.
    - Carica e modifica l'XML (iniezione siti + sensori LiDAR).
    - Espone model/data e gli ID utili (agent body, lidar).
    - Viewer opzionale (render_mode="human").
    - Reset base: gestisce solo il seeding; le subclass implementano la logica.
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 30}

    def __init__(
        self,
        num_rays: int = 108,
        training: bool = True,
        render_mode: str | None = None,
        model_path: str | None = None,
    ) -> None:
        super().__init__()

        # --- Paths coerenti alla nuova struttura ---
        this_file = Path(__file__).resolve()
        self.SRC_DIR = this_file.parents[1]     # <ROOT>/src
        self.ROOT_DIR = this_file.parents[2]    # <ROOT>
        self.ASSETS = self.ROOT_DIR / "assets"

        self.training = bool(training)
        self.render_mode = render_mode
        self.num_rays = int(num_rays)

        # Default model: assets/world.xml
        self.model_path = str(self.ASSETS / "world.xml") if model_path is None else model_path

        # Forza JAX su CPU se una subclass lo usa (non fa danni se assente)
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

        # --- Costruzione modello MuJoCo ---
        self.xml_model = self.load_and_modify_xml_model()
        self.model = mujoco.MjModel.from_xml_string(self.xml_model)
        self.data = mujoco.MjData(self.model)

        # --- IDs utili ---
        self.mobile_robot_ID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "agent_body")
        if self.mobile_robot_ID < 0:
            raise ValueError("Body 'agent_body' non trovato nel modello MuJoCo.")

        # Lidar sensor IDs (li abbiamo iniettati qui sotto)
        self.lidar_sensor_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
            for i in range(self.num_rays)
        ]
        if any(i < 0 for i in self.lidar_sensor_ids):
            missing = [k for k, i in enumerate(self.lidar_sensor_ids) if i < 0]
            raise RuntimeError(f"Sensori LiDAR mancanti: indici {missing}")

        # Viewer opzionale
        self.viewer = None
        if self.render_mode == "human":
            self._setup_viewer()

        # Nota: non chiamiamo reset qui; le subclass gestiscono stato e observation space

    # ------------------------------------------------------------------
    # XML loading & LiDAR injection
    # ------------------------------------------------------------------
    def load_and_modify_xml_model(self) -> str:
        """
        Carica il modello XML e inietta:
        - <site> 'lidar_site_i' su 'agent_body'
        - <rangefinder> 'lidar_i' associati ai site
        NB: non duplica se già presenti: prima pulisce eventuali 'rangefinder' esistenti.
        """
        xml_path = Path(self.model_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"Model file non trovato: {xml_path}")

        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        # Trova il body dell'agente
        mobile_robot_body = None
        for body in root.findall(".//body"):
            if body.get("name") == "agent_body":
                mobile_robot_body = body
                break
        if mobile_robot_body is None:
            raise ValueError("Body 'agent_body' non trovato nell'XML.")

        # Trova o crea <sensor>
        sensor_tag = root.find(".//sensor")
        if sensor_tag is None:
            sensor_tag = ET.SubElement(root, "sensor")

        # Rimuove eventuali rangefinder esistenti (evita duplicazioni)
        for rf in list(sensor_tag.findall("rangefinder")):
            sensor_tag.remove(rf)

        # Opzionale: rimuovi anche vecchi site lidar_*
        old_sites = [s for s in mobile_robot_body.findall("site") if (s.get("name") or "").startswith("lidar_site_")]
        for s in old_sites:
            mobile_robot_body.remove(s)

        # Aggiunge siti + sensori LiDAR
        # pos = 1 mm dal centro per evitare self-collisions; zaxis = direzione del raggio
        for i in range(self.num_rays):
            angle = (i / self.num_rays) * 2 * np.pi
            angle = (angle + np.pi) % (2 * np.pi) - np.pi  # normalizza in [-pi, pi]

            site = ET.SubElement(mobile_robot_body, "site", {
                "name": f"lidar_site_{i}",
                "pos": f"{0.001*np.cos(angle):.6f} {0.001*np.sin(angle):.6f} 0",
                "size": "0.01",
                "rgba": "1 0 0 0.3",
                "zaxis": f"{np.cos(angle):.6f} {np.sin(angle):.6f} 0",
            })

            ET.SubElement(sensor_tag, "rangefinder", {
                "name": f"lidar_{i}",
                "site": f"lidar_site_{i}",
            })

        return ET.tostring(root, encoding="unicode")

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------
    def _setup_viewer(self) -> None:
        # Viewer passivo (non blocca il thread principale)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # Camera top-down di default
        self.viewer.cam.distance = 25.0
        self.viewer.cam.azimuth = 0.0
        self.viewer.cam.elevation = -90.0
        self.viewer.cam.lookat[:] = [0, 0, 1]

    def render(self) -> bool:
        if self.render_mode == "human":
            if self.viewer is None:
                self._setup_viewer()
            self.viewer.sync()
            return True
        return False

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # ------------------------------------------------------------------
    # Gymnasium API (base)
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Base reset: gestisce solo il seeding PRNG.
        Le subclass DEVONO restituire (obs, info) con la loro logica e possono chiamare questo metodo.
        """
        if seed is not None:
            try:
                # Gymnasium seeding helper (se disponibile)
                import random
                np.random.seed(seed)
                random.seed(seed)
            except Exception:
                pass
        # Non ritorniamo observation qui: le subclass lo faranno.
        # È comunque safe ignorare il valore di ritorno quando chiamato da subclass.
        return None

    def step(self, action):
        """
        Non implementato: le subclass devono definire step(observation, reward, terminated, truncated, info).
        """
        raise NotImplementedError("La subclass deve implementare 'step'.")
