# Joint and Actuator Names

## Legs

### Left Leg
- `hip_pitch_left` - Left hip rotation (pitch)
- `hip_roll_left` - Left hip rotation (roll)
- `hip_yaw_left` - Left hip rotation (yaw)
- `knee_pitch_left` - Left knee rotation (pitch)
- `ankle_pitch_left` - Left ankle rotation (pitch)
- `ankle_roll_left` - Left ankle rotation (roll)

### Right Leg
- `hip_pitch_right` - Right hip rotation (pitch)
- `hip_roll_right` - Right hip rotation (roll)
- `hip_yaw_right` - Right hip rotation (yaw)
- `knee_pitch_right` - Right knee rotation (pitch)
- `ankle_pitch_right` - Right ankle rotation (pitch)
- `ankle_roll_right` - Right ankle rotation (roll)

## Arms

### Left Arm
- `shoulder_pitch_left` - Left shoulder rotation (pitch)
- `shoulder_roll_left` - Left shoulder rotation (roll)
- `shoulder_yaw_left` - Left shoulder rotation (yaw)
- `elbow_pitch_left` - Left elbow rotation (pitch)
- `wrist_pitch_left` - Left wrist rotation (pitch)
- `wrist_roll_left` - Left wrist rotation (roll)

### Right Arm
- `shoulder_pitch_right` - Right shoulder rotation (pitch)
- `shoulder_roll_right` - Right shoulder rotation (roll)
- `shoulder_yaw_right` - Right shoulder rotation (yaw)
- `elbow_pitch_right` - Right elbow rotation (pitch)
- `wrist_pitch_right` - Right wrist rotation (pitch)
- `wrist_roll_right` - Right wrist rotation (roll)

## Torso
- `torso_pitch` - Spine rotation (pitch)
- `torso_roll` - Spine rotation (roll)
- `torso_yaw` - Spine rotation (yaw)

## Head
- `head_pitch` - Head rotation (pitch)
- `head_roll` - Head rotation (roll)
- `head_yaw` - Head rotation (yaw)

---

# Actuator Names

For most joints, the actuator name is the same as the joint name (1:1 mapping). The ankle joints are an exception due to the differential mechanism.

## Ankle Differential Mechanism

The ankle uses two motors (lower, upper) that together produce pitch and roll motion. The mapping between joint space and actuator space requires kinematics transformations (FK/IK).

| Joint Name | Actuator Name | Motor |
|-----------|---------------|-------|
| `ankle_pitch_left` | `ankle_lower_left` | Lower motor (q_lower) |
| `ankle_roll_left` | `ankle_upper_left` | Upper motor (q_upper) |
| `ankle_pitch_right` | `ankle_lower_right` | Lower motor (q_lower) |
| `ankle_roll_right` | `ankle_upper_right` | Upper motor (q_upper) |

## All Other Joints

All non-ankle joints have a direct 1:1 mapping where the actuator name equals the joint name (e.g., `hip_pitch_left` in joint space = `hip_pitch_left` in actuator space).
