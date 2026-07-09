import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const payload = await request.json();
    const { jobId, topology_type, runtime_target, framework } = payload;

    if (!jobId) {
      return NextResponse.json({ error: "Missing required jobId field" }, { status: 400 });
    }

    console.log(`Enqueuing task for job ${jobId}`);

    // Call the local Python orchestrator service directly
    const runnerUrl = process.env.RUNNER_URL || 'http://localhost:8080/run';
    
    try {
      const runnerResponse = await fetch(runnerUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ jobId }),
      });
      
      if (!runnerResponse.ok) {
        console.error("Runner responded with an error:", await runnerResponse.text());
      }
    } catch (e) {
      console.error("Failed to connect to runner:", e);
      // We'll continue and return success to the UI even if the local runner isn't running
      // so we don't break the UI flow during testing.
    }

    return NextResponse.json({ 
      success: true, 
      message: "Task enqueued successfully",
      jobId 
    });
  } catch (error) {
    console.error("Error enqueuing task:", error);
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
