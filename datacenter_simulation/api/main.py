"""
FastAPI application for Data Center & Cloud Services Consulting Simulation
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn

from ..run_simulation import DataCenterSimulation

app = FastAPI(
    title="Data Center & Cloud Services Consulting Simulation API",
    description="API for simulating data center infrastructure and operations from 2025-2040",
    version="1.0.0"
)

class SimulationRequest(BaseModel):
    """Request model for simulation parameters"""
    years: int = 15
    regions: List[str]
    initial_workload: float
    facility_size: float

class SimulationResponse(BaseModel):
    """Response model for simulation results"""
    recommendations: List[Dict]
    infrastructure: Dict[str, float]
    strategies: List[str]

@app.post("/simulate", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    """
    Run the data center simulation with specified parameters
    
    Args:
        request: Simulation parameters
        
    Returns:
        Simulation results including site recommendations and strategies
    """
    try:
        # Initialize simulation
        simulation = DataCenterSimulation()
        
        # Run simulation
        results = simulation.run_simulation(
            years=request.years,
            regions=request.regions,
            initial_workload=request.initial_workload,
            facility_size=request.facility_size
        )
        
        # Return key results
        return SimulationResponse(
            recommendations=results['recommendations'],
            infrastructure=results['infrastructure'],
            strategies=results['strategies']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

def start():
    """Start the API server"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start() 