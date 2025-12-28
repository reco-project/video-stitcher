import React, { createContext, useState, useContext } from 'react';

const CameraContext = createContext(null);

export function CameraProvider({ children }) {
	const [yawRange, setYawRange] = useState(140); // horizontal panning in degrees
	const [pitchRange, setPitchRange] = useState(20); // vertical panning in degrees

	return (
		<CameraContext.Provider value={{ yawRange, setYawRange, pitchRange, setPitchRange }}>
			{children}
		</CameraContext.Provider>
	);
}

export function useCameraControls() {
	const context = useContext(CameraContext);
	if (!context) {
		throw new Error('useCameraControls must be used within CameraProvider');
	}
	return context;
}
