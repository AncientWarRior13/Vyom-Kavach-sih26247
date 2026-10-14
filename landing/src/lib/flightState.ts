import * as THREE from 'three'

export interface FlightState {
  progress: number
  damped: number
  speed: number
  mouseX: number
  mouseY: number
  mouseActive: boolean
  droneWorld: THREE.Vector3
  droneForward: THREE.Vector3
}

export const flight: FlightState = {
  progress: 0,
  damped: 0,
  speed: 0,
  mouseX: 0,
  mouseY: 0,
  mouseActive: false,
  droneWorld: new THREE.Vector3(0, 25, 0),
  droneForward: new THREE.Vector3(0, 0, -1),
}
