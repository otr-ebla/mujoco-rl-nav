import numpy as np
import mujoco

COLLISION_THRESHOLD = 0.4  # Default threshold for collision detection

class CollisionDetector:
    """Enhanced collision detection for the HAMRRLN environment"""
    def __init__(self, model, data, robot_body_name = "agent_body", human_body_names = None):
        self.model = model
        self.data = data 

        self.robot_body_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_body_name)

        if human_body_names is None:
            human_body_names = [f"human{i+1}" for i in range(5)]  # Default to 5 humans if not specified
        
        self.human_body_ids = []
        for name in human_body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                self.human_body_ids.append(body_id)

        # Cache obstacle/walls geometry IDs
        self.obstacle_geom_ids = []
        for i in range(self.model.ngeom):
            geom_bodyid = self.model.geom_bodyid[i]
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            # Exclude robot, human, and target sphere
            if (geom_bodyid != self.robot_body_ID and 
                geom_bodyid not in self.human_body_ids and 
                geom_name != "sphere" and
                geom_name != "floor"):
                self.obstacle_geom_ids.append(i)
            
        self._cache_geom_ids()

    def _cache_geom_ids(self):
        """Cache the geometry IDs for the robot and humans."""
        self.robot_geom_ids = []
        self.human_geom_ids = []
       
        for i in range(self.model.ngeom):
            if self.model.geom_bodyid[i] == self.robot_body_ID:
                self.robot_geom_ids.append(i)   
        
        for human_id in self.human_body_ids:
            for i in range(self.model.ngeom):
                if self.model.geom_bodyid[i] == human_id:
                    self.human_geom_ids.append(i)

    def check_robot_obstacle_collision(self, collision_threshold=COLLISION_THRESHOLD):
        """
        Check for collisions between the robot and obstacles.
        Returns: (collision_detected, contact_info_list)
        """
        collisions = []
        
        for robot_geom_id in self.robot_geom_ids:
            for obstacle_geom_id in self.obstacle_geom_ids:
                # Calculate distance between geometries
                # Fixed: Added distmax parameter and fromto array
                fromto = np.zeros(6, dtype=np.float64)  # Will store closest points
                distance = mujoco.mj_geomDistance(
                    self.model, self.data, robot_geom_id, obstacle_geom_id, 
                    collision_threshold * 2.0,  # distmax - maximum distance to compute
                    fromto  # array to store closest points
                )
                
                if distance < collision_threshold:
                    collision_info = {
                        'robot_geom_id': robot_geom_id,
                        'obstacle_geom_id': obstacle_geom_id,
                        'distance': distance,
                        'robot_geom_name': mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom_id),
                        'obstacle_geom_name': mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, obstacle_geom_id),
                        'closest_points': fromto.copy()  # Store the closest points
                    }
                    collisions.append(collision_info)
        
        return len(collisions) > 0, collisions
    
    def check_robot_human_collision_contacts(self):
        """
        Method 1: Check for collisions using mjData.contact
        Returns: (collision_detected, contact_info_list)
        """
        collisions = []
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            # Check if this contact involves robot and human
            robot_involved = geom1_id in self.robot_geom_ids or geom2_id in self.robot_geom_ids
            human_involved = geom1_id in self.human_geom_ids or geom2_id in self.human_geom_ids
            
            if robot_involved and human_involved:
                # Get contact details
                contact_info = {
                    'contact_id': i,
                    'geom1_id': geom1_id,
                    'geom2_id': geom2_id,
                    'position': contact.pos.copy(),
                    'normal': contact.frame[:3].copy(),  # Contact normal
                    'distance': contact.dist,  # Penetration depth (negative if penetrating)
                    'force': np.linalg.norm(contact.lambda_[:3]) if hasattr(contact, 'lambda_') else 0
                }
                collisions.append(contact_info)
        
        return len(collisions) > 0, collisions
    
    def check_robot_human_collision_distance(self, collision_threshold=0.4):
        """
        Method 2: Check for collisions using distance-based detection
        Returns: (collision_detected, collision_info_list)
        """
        collisions = []
        robot_pos = self.data.xpos[self.robot_body_ID]
        
        for human_id in self.human_body_ids:
            human_pos = self.data.xpos[human_id]
            distance = np.linalg.norm(robot_pos[:2] - human_pos[:2])  # 2D distance
            
            if distance < collision_threshold:
                collision_info = {
                    'human_body_id': human_id,
                    'distance': distance,
                    'robot_pos': robot_pos.copy(),
                    'human_pos': human_pos.copy()
                }
                collisions.append(collision_info)
        
        return len(collisions) > 0, collisions
    
    def check_robot_human_collision_geom_distance(self, collision_threshold=0.4):
        """
        Method 3: More precise collision detection using mj_geomDistance
        Returns: (collision_detected, collision_info_list)
        """
        collisions = []
        
        for robot_geom_id in self.robot_geom_ids:
            for human_geom_id in self.human_geom_ids:
                # Calculate distance between geometries
                # Fixed: Added distmax parameter and fromto array
                fromto = np.zeros(6, dtype=np.float64)  # Will store closest points
                distance = mujoco.mj_geomDistance(
                    self.model, self.data, robot_geom_id, human_geom_id,
                    collision_threshold * 2.0,  # distmax - maximum distance to compute
                    fromto  # array to store closest points
                )
                
                if distance < collision_threshold:
                    collision_info = {
                        'robot_geom_id': robot_geom_id,
                        'human_geom_id': human_geom_id,
                        'distance': distance,
                        'robot_geom_name': mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom_id),
                        'human_geom_name': mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, human_geom_id),
                        'closest_points': fromto.copy()  # Store the closest points
                    }
                    collisions.append(collision_info)
        
        return len(collisions) > 0, collisions