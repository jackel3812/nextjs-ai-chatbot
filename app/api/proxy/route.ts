import { NextRequest, NextResponse } from 'next/server';

export const maxDuration = 60;

/**
 * This proxy route allows the Next.js frontend to communicate with the Python backend
 * using the same origin, avoiding CORS issues in deployment.
 */
export async function POST(request: NextRequest) {
  try {
    // Get the backend URL from environment variable or use default
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5000';
    
    // Read the request body
    const body = await request.json();
    
    // Get the path from the query parameter
    const { searchParams } = new URL(request.url);
    const path = searchParams.get('path') || '/api/chat';
    
    // Forward the request to the Python backend
    const response = await fetch(`${backendUrl}${path}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    
    // Check if the response is OK
    if (!response.ok) {
      console.error(`Backend error: ${response.status} ${response.statusText}`);
      return NextResponse.json(
        { error: 'Error communicating with backend service' },
        { status: response.status }
      );
    }
    
    // Return the response from the Python backend
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Proxy error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    // Get the backend URL from environment variable or use default
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5000';
    
    // Get the path from the query parameter
    const { searchParams } = new URL(request.url);
    const path = searchParams.get('path') || '/';
    
    // Forward the request to the Python backend
    const response = await fetch(`${backendUrl}${path}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    // Check if the response is OK
    if (!response.ok) {
      console.error(`Backend error: ${response.status} ${response.statusText}`);
      return NextResponse.json(
        { error: 'Error communicating with backend service' },
        { status: response.status }
      );
    }
    
    // Return the response from the Python backend
    const data = await response.text();
    return new NextResponse(data, {
      status: response.status,
      headers: {
        'Content-Type': response.headers.get('Content-Type') || 'text/plain',
      },
    });
  } catch (error) {
    console.error('Proxy error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
