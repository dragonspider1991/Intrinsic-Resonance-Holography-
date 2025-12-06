/**
 * 3D Visualization Component
 * Renders network topology and eigenvalue spectrum using Three.js
 */

import React, { useRef, useEffect, useState } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { useAppStore } from '../store/appStore';
import { apiClient } from '../services/api';

export const Visualization3D: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  const { networkConfig, ui } = useAppStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(5, 5, 5);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);

    // Orbit Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 2;
    controls.maxDistance = 50;
    controlsRef.current = controls;

    // Animation loop
    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
      if (!containerRef.current) return;
      const newWidth = containerRef.current.clientWidth;
      const newHeight = containerRef.current.clientHeight;
      
      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      controls.dispose();
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, []);

  // Load and render visualization data
  useEffect(() => {
    const loadVisualization = async () => {
      if (!sceneRef.current) return;

      setLoading(true);
      setError(null);

      try {
        // Clear previous objects (keep lights)
        const objectsToRemove: THREE.Object3D[] = [];
        sceneRef.current.traverse((obj) => {
          if (obj instanceof THREE.Mesh || obj instanceof THREE.Line) {
            objectsToRemove.push(obj);
          }
        });
        objectsToRemove.forEach((obj) => sceneRef.current!.remove(obj));

        if (ui.visualizationType === 'network' || ui.visualizationType === 'both') {
          // Load network 3D data
          const networkData = await apiClient.getNetwork3D(networkConfig);

          // Render nodes
          const nodeGeometry = new THREE.SphereGeometry(0.1, 16, 16);
          networkData.nodes.forEach((node) => {
            const material = new THREE.MeshStandardMaterial({
              color: node.color,
              metalness: 0.3,
              roughness: 0.7,
            });
            const sphere = new THREE.Mesh(nodeGeometry, material);
            sphere.position.set(node.position[0], node.position[1], node.position[2]);
            sphere.scale.setScalar(node.size);
            sceneRef.current!.add(sphere);
          });

          // Render edges
          networkData.edges.forEach((edge) => {
            const sourceNode = networkData.nodes[edge.source];
            const targetNode = networkData.nodes[edge.target];
            
            const points = [
              new THREE.Vector3(...sourceNode.position),
              new THREE.Vector3(...targetNode.position),
            ];
            const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
            const lineMaterial = new THREE.LineBasicMaterial({
              color: edge.color,
              opacity: edge.opacity,
              transparent: true,
            });
            const line = new THREE.Line(lineGeometry, lineMaterial);
            sceneRef.current!.add(line);
          });
        }

        if (ui.visualizationType === 'spectrum') {
          // Load spectrum 3D data
          const spectrumData = await apiClient.getSpectrum3D(networkConfig);

          // Render points
          const pointGeometry = new THREE.SphereGeometry(0.05, 8, 8);
          spectrumData.points.forEach((point) => {
            const material = new THREE.MeshStandardMaterial({
              color: point.color,
              metalness: 0.3,
              roughness: 0.7,
            });
            const sphere = new THREE.Mesh(pointGeometry, material);
            sphere.position.set(point.x, point.y, point.z);
            sphere.scale.setScalar(point.size);
            sceneRef.current!.add(sphere);
          });
        }

        setLoading(false);
      } catch (err: any) {
        console.error('Error loading 3D visualization:', err);
        setError(err.message || 'Failed to load visualization');
        setLoading(false);
      }
    };

    loadVisualization();
  }, [networkConfig, ui.visualizationType]);

  return (
    <Box
      ref={containerRef}
      sx={{
        width: '100%',
        height: '100%',
        position: 'relative',
        borderRadius: 2,
        overflow: 'hidden',
        bgcolor: 'background.default',
      }}
    >
      {loading && (
        <Box
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 2,
          }}
        >
          <CircularProgress />
          <Typography variant="body2" color="text.secondary">
            Loading 3D visualization...
          </Typography>
        </Box>
      )}
      {error && (
        <Box
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            textAlign: 'center',
          }}
        >
          <Typography variant="body1" color="error">
            {error}
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default Visualization3D;
