"use client";

import { useState, useEffect } from "react";
import { collection, query, orderBy, onSnapshot, limit } from "firebase/firestore";
import { db } from "@/services/firebase";
import { Job } from "@/types";

export const useJobs = (maxCount = 24) => {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const q = query(
      collection(db, "jobs"),
      orderBy("createdAt", "desc"),
      limit(maxCount)
    );

    const unsubscribe = onSnapshot(
      q,
      (snapshot) => {
        const jobsData = snapshot.docs.map((doc) => ({
          id: doc.id,
          ...doc.data(),
        })) as Job[];
        setJobs(jobsData);
        setLoading(false);
      },
      (err) => {
        console.error("Error fetching jobs:", err);
        setError(err);
        setLoading(false);
      }
    );

    return () => unsubscribe();
  }, [maxCount]);

  return { jobs, loading, error };
};
