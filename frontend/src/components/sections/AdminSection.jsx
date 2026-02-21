import React, { useCallback, useEffect, useState } from "react";
import {
  CheckCircle2,
  Loader2,
  PauseCircle,
  PlayCircle,
  ShieldCheck,
  UserX,
  Users
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { adminApi } from "@/services/api";
import { toast } from "sonner";

export const AdminSection = ({ user }) => {
  const [loading, setLoading] = useState(true);
  const [processingId, setProcessingId] = useState(null);
  const [statusFilter, setStatusFilter] = useState("all");
  const [pendingClinicians, setPendingClinicians] = useState([]);
  const [clinicians, setClinicians] = useState([]);
  const [auditLogs, setAuditLogs] = useState([]);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [pending, allClinicians, logs] = await Promise.all([
        adminApi.getPendingClinicians().catch(() => []),
        adminApi.getClinicians(statusFilter).catch(() => []),
        adminApi.getAudit(80).catch(() => [])
      ]);
      setPendingClinicians(pending || []);
      setClinicians(allClinicians || []);
      setAuditLogs(logs || []);
    } catch (error) {
      toast.error("Could not load admin data.");
    } finally {
      setLoading(false);
    }
  }, [statusFilter]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const runAdminAction = async (userId, action, label) => {
    setProcessingId(userId + action);
    try {
      const notes = window.prompt(`Optional note for ${label}:`, "") || "";
      if (action === "approve") {
        await adminApi.approveClinician(userId, notes);
      } else if (action === "reject") {
        await adminApi.rejectClinician(userId, notes);
      } else if (action === "suspend") {
        await adminApi.suspendClinician(userId, notes);
      } else if (action === "reactivate") {
        await adminApi.reactivateClinician(userId, notes);
      }
      toast.success(`${label} completed.`);
      loadData();
    } catch (error) {
      toast.error(error.message || `${label} failed.`);
    } finally {
      setProcessingId(null);
    }
  };

  if (user?.role !== "admin") {
    return (
      <section className="py-10">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <Card className="border-2 border-border">
            <CardContent className="p-8 text-center text-muted-foreground">
              Admin access required.
            </CardContent>
          </Card>
        </div>
      </section>
    );
  }

  return (
    <section className="py-8 sm:py-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 mb-4">
            <ShieldCheck className="h-5 w-5 text-primary" />
            <span className="text-base font-medium text-primary">Admin Governance</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl font-bold text-foreground mb-2">
            Clinician Access Control
          </h2>
          <p className="text-accessible text-muted-foreground">
            Review, approve, reject, suspend, and reactivate clinician accounts.
          </p>
        </div>

        <div className="grid gap-4 md:grid-cols-3 mb-8">
          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="text-sm">Pending Clinicians</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold text-amber-600">{pendingClinicians.length}</p>
            </CardContent>
          </Card>
          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="text-sm">Total Clinicians</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold">{clinicians.length}</p>
            </CardContent>
          </Card>
          <Card className="border-2 border-border">
            <CardHeader>
              <CardTitle className="text-sm">Audit Entries</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold">{auditLogs.length}</p>
            </CardContent>
          </Card>
        </div>

        <Card className="border-2 border-border mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5 text-primary" />
              Pending Approval Queue
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {loading ? (
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
            ) : pendingClinicians.length === 0 ? (
              <p className="text-sm text-muted-foreground">No pending clinicians.</p>
            ) : (
              pendingClinicians.map((clinician) => (
                <div key={clinician.user_id} className="rounded-lg border p-3 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                  <div>
                    <p className="font-medium">{clinician.name}</p>
                    <p className="text-xs text-muted-foreground">{clinician.email}</p>
                    <p className="text-xs text-muted-foreground">
                      Requested: {clinician.requested_role || clinician.role} | Created: {new Date(clinician.created_at).toLocaleString()}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      className="gap-1"
                      onClick={() => runAdminAction(clinician.user_id, "approve", "Approve clinician")}
                      disabled={processingId === clinician.user_id + "approve"}
                    >
                      {processingId === clinician.user_id + "approve" ? <Loader2 className="h-3 w-3 animate-spin" /> : <CheckCircle2 className="h-3 w-3" />}
                      Approve
                    </Button>
                    <Button
                      size="sm"
                      variant="destructive"
                      className="gap-1"
                      onClick={() => runAdminAction(clinician.user_id, "reject", "Reject clinician")}
                      disabled={processingId === clinician.user_id + "reject"}
                    >
                      {processingId === clinician.user_id + "reject" ? <Loader2 className="h-3 w-3 animate-spin" /> : <UserX className="h-3 w-3" />}
                      Reject
                    </Button>
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>

        <Card className="border-2 border-border mb-8">
          <CardHeader>
            <CardTitle>Clinician Directory</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div>
              <label className="text-sm text-muted-foreground">Filter</label>
              <select
                className="w-full mt-2 rounded-md border border-input bg-background px-3 py-2"
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
              >
                <option value="all">All</option>
                <option value="pending">Pending</option>
                <option value="approved">Approved</option>
                <option value="rejected">Rejected</option>
                <option value="suspended">Suspended</option>
              </select>
            </div>

            {loading ? (
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
            ) : clinicians.length === 0 ? (
              <p className="text-sm text-muted-foreground">No clinicians for this filter.</p>
            ) : (
              clinicians.map((clinician) => (
                <div key={clinician.user_id} className="rounded-lg border p-3 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                  <div>
                    <p className="font-medium">{clinician.name}</p>
                    <p className="text-xs text-muted-foreground">{clinician.email}</p>
                    <p className="text-xs text-muted-foreground">
                      Approval: {clinician.clinician_approval_status || "not_applicable"} | Account: {clinician.account_status || "active"}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    {clinician.account_status !== "suspended" ? (
                      <Button
                        size="sm"
                        variant="outline"
                        className="gap-1"
                        onClick={() => runAdminAction(clinician.user_id, "suspend", "Suspend clinician")}
                        disabled={processingId === clinician.user_id + "suspend"}
                      >
                        {processingId === clinician.user_id + "suspend" ? <Loader2 className="h-3 w-3 animate-spin" /> : <PauseCircle className="h-3 w-3" />}
                        Suspend
                      </Button>
                    ) : (
                      <Button
                        size="sm"
                        variant="outline"
                        className="gap-1"
                        onClick={() => runAdminAction(clinician.user_id, "reactivate", "Reactivate clinician")}
                        disabled={processingId === clinician.user_id + "reactivate"}
                      >
                        {processingId === clinician.user_id + "reactivate" ? <Loader2 className="h-3 w-3 animate-spin" /> : <PlayCircle className="h-3 w-3" />}
                        Reactivate
                      </Button>
                    )}
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>

        <Card className="border-2 border-border">
          <CardHeader>
            <CardTitle>Recent Admin Audit</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {loading ? (
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
            ) : auditLogs.length === 0 ? (
              <p className="text-sm text-muted-foreground">No audit entries yet.</p>
            ) : (
              auditLogs.slice(0, 20).map((log) => (
                <div key={log.id} className="rounded-lg border p-2">
                  <p className="text-sm font-medium">{log.action}</p>
                  <p className="text-xs text-muted-foreground">
                    Admin: {log.admin_name} | Target: {log.target_user_id} | {new Date(log.created_at).toLocaleString()}
                  </p>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </div>
    </section>
  );
};

export default AdminSection;
