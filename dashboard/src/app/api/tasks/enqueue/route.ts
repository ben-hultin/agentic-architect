import { NextResponse } from "next/server";
// import { CloudTasksClient } from "@google-cloud/tasks";

export async function POST(request: Request) {
  try {
    const payload = await request.json();
    const { jobId, framework, evalSetPath } = payload;

    if (!jobId || !framework) {
      return NextResponse.json({ error: "Missing required fields" }, { status: 400 });
    }

    console.log(`Enqueuing task for job ${jobId} with framework ${framework}`);

    // In a real implementation:
    /*
    const client = new CloudTasksClient();
    const project = process.env.GCP_PROJECT_ID;
    const queue = process.env.GCP_QUEUE_NAME;
    const location = process.env.GCP_LOCATION;
    const url = process.env.RUNNER_URL; // URL of the Cloud Run runner

    const parent = client.queuePath(project, location, queue);

    const task = {
      httpRequest: {
        httpMethod: 'POST',
        url,
        body: Buffer.from(JSON.stringify(payload)).toString('base64'),
        headers: {
          'Content-Type': 'application/json',
        },
      },
    };

    const [response] = await client.createTask({ parent, task });
    console.log(`Created task ${response.name}`);
    */

    return NextResponse.json({ 
      success: true, 
      message: "Task enqueued successfully (Mock)",
      jobId 
    });
  } catch (error) {
    console.error("Error enqueuing task:", error);
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
