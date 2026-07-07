"use client";

import { useState, useEffect } from "react";
import { collection, query, where, orderBy, onSnapshot } from "firebase/firestore";
import { db } from "@/services/firebase";
import { Report } from "@/types";

export const useReports = (jobId?: string) => {
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let q = query(collection(db, "reports"), orderBy("timestamp", "desc"));

    if (jobId) {
      q = query(
        collection(db, "reports"),
        where("jobId", "==", jobId),
        orderBy("timestamp", "desc")
      );
    }

    const unsubscribe = onSnapshot(
      q,
      (snapshot) => {
        const reportsData = snapshot.docs.map((doc) => ({
          id: doc.id,
          ...doc.data(),
        })) as Report[];
        setReports(reportsData);
        setLoading(false);
      },
      (err) => {
        console.error("Error fetching reports:", err);
        setError(err);
        setLoading(false);
      }
    );

    return () => unsubscribe();
  }, [jobId]);

  return { reports, loading, error };
};
