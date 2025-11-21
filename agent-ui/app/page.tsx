// app/page.tsx (client)
"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import Image from "next/image";

const GOOGLE_CLIENT_ID = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID ?? "";

/** load GSI script once */
function loadGoogleScript() {
  if (typeof window === "undefined") return;
  if (document.getElementById("google-identity")) return;
  const s = document.createElement("script");
  s.src = "https://accounts.google.com/gsi/client";
  s.id = "google-identity";
  s.async = true;
  s.defer = true;
  document.head.appendChild(s);
}

/** message shape */
type Msg = { sender: "user" | "bot"; text: string };

export default function ChatPage() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const inflightRef = useRef<AbortController | null>(null);

  // Demo fallback token — you can remove this if you always use google auth.
  // For prod don't ship fallback tokens!!
  const DEMO_TOKEN =
    "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InRlc3RAZXhhbXBsZS5jb20iLCJnaXRodWJJZCI6IjEyMyJ9.1Wdro2KrI8fA7tr2tF3g4RY41_cmMjFKFZryU9B30";

  // JWT returned by backend after exchanging google id_token
  const [jwtToken, setJwtToken] = useState<string | null>(null);
  const [googleUser, setGoogleUser] = useState<{ email?: string; name?: string } | null>(null);

  // auto-scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  // initialize Google Identity
  useEffect(() => {
    if (!GOOGLE_CLIENT_ID) {
      console.warn("NEXT_PUBLIC_GOOGLE_CLIENT_ID not set. Google sign-in disabled.");
      return;
    }

    loadGoogleScript();

    let didInit = false;
    const tryInit = () => {
      // wait until google object is available
      if (didInit) return;
      const w = window as any;
      if (!w.google || !w.google.accounts || !w.google.accounts.id) return;

      try {
        w.google.accounts.id.initialize({
          client_id: GOOGLE_CLIENT_ID,
          callback: async (resp: any) => {
            const id_token = resp?.credential;
            if (!id_token) {
              console.error("Google callback did not return credential");
              return;
            }

            try {
              // Exchange id_token for your app JWT
              const r = await fetch("http://127.0.0.1:8000/api/auth/google/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ id_token }),
              });

              const data = await r.json();
              if (r.ok && data.jwt) {
                setJwtToken(data.jwt);
                // prefer name/email returned by backend; fallback to id_token payload
                if (data.email || data.name) {
                  setGoogleUser({ email: data.email, name: data.name });
                } else {
                  try {
                    const payload = JSON.parse(atob(id_token.split(".")[1]));
                    setGoogleUser({ email: payload.email, name: payload.name });
                  } catch {
                    // ignore
                  }
                }
              } else {
                console.error("Google auth exchange failed:", data);
              }
            } catch (err) {
              console.error("Network error exchanging id_token:", err);
            }
          },
        });

        // Render the button into the container. The container may be shown/hidden.
        const el = document.getElementById("g_id_signin");
        if (el) {
          w.google.accounts.id.renderButton(el, {
            theme: "outline",
            size: "large",
            type: "standard",
            shape: "rectangular",
          });
        }

        // Optional: prompt for One-tap (comment out if not desired)
        // w.google.accounts.id.prompt();

        didInit = true;
      } catch (e) {
        console.warn("Error initializing Google Identity:", e);
      }
    };

    const id = window.setInterval(() => tryInit(), 200);
    // try init immediately too
    tryInit();

    return () => {
      clearInterval(id);
    };
  }, []);

  async function sendMessage() {
    const trimmed = input.trim();
    if (!trimmed || isTyping) return;

    setMessages((p) => [...p, { sender: "user", text: trimmed }]);
    setInput("");

    const ac = new AbortController();
    inflightRef.current = ac;
    setIsTyping(true);

    const url = threadId ? `http://127.0.0.1:8000/api/chat/?threadId=${threadId}` : `http://127.0.0.1:8000/api/chat/`;

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: jwtToken ? `Bearer ${jwtToken}` : DEMO_TOKEN, // use jwtToken when available
        },
        body: JSON.stringify({ message: trimmed }),
        signal: ac.signal,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`HTTP ${res.status}: ${text}`);
      }

      const data = await res.json();
      if (data.threadId) setThreadId(data.threadId);

      const botText: string = (data.response ?? data.agent_raw) || "[no response]";
      setMessages((p) => [...p, { sender: "bot", text: botText }]);
    } catch (err: any) {
      if (err.name === "AbortError") {
        setMessages((p) => [...p, { sender: "bot", text: "*Response interrupted.*" }]);
      } else {
        setMessages((p) => [...p, { sender: "bot", text: `*Error:* ${String(err)}` }]);
      }
    } finally {
      inflightRef.current = null;
      setIsTyping(false);
    }
  }

  function stopResponse() {
    if (inflightRef.current) inflightRef.current.abort();
  }

  function signOutGoogle() {
    setJwtToken(null);
    setGoogleUser(null);
    // disable auto-select
    try {
      (window as any).google?.accounts?.id?.disableAutoSelect?.();
    } catch {
      /* ignore */
    }
  }

  // Markdown overrides - links open in new tab and styled blue
  const markdownComponents: any = {
    a: ({ node, ...props }: any) => (
      <a
        {...props}
        href={props.href}
        target="_blank"
        rel="noopener noreferrer"
        title="Click to follow"
        className="text-blue-600 hover:underline hover:text-blue-700"
      />
    ),
    p: (props: any) => <p className="mb-2 leading-relaxed">{props.children}</p>,
    li: (props: any) => <li className="ml-4 list-disc">{props.children}</li>,
    code: (props: any) => <code className="bg-gray-100 text-gray-900 px-1 rounded text-sm">{props.children}</code>,
    pre: (props: any) => <pre className="bg-gray-900 text-gray-50 p-3 rounded overflow-auto">{props.children}</pre>,
  };

  return (
    <div className="flex flex-col items-center justify-start min-h-screen bg-gray-50 text-gray-900 p-6">
      {/* HEADER */}
      <div className="w-full max-w-3xl flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 relative rounded-md bg-white shadow flex items-center justify-center">
            <Image src="/gojeklogo.png" alt="Logo" width={42} height={42} className="rounded" />
          </div>
          <h1 className="text-3xl font-bold tracking-tight">Support Chat</h1>
        </div>

        <div className="flex items-center gap-3">
          {!jwtToken ? (
            // container that Google will render into. Hidden if no client id.
            <div id="g_id_signin">{!GOOGLE_CLIENT_ID && <span className="text-sm text-red-500">GOOGLE_CLIENT_ID not set</span>}</div>
          ) : (
            <div className="flex items-center gap-3">
              <div className="text-sm text-gray-700">Signed in as <strong>{googleUser?.email ?? "user"}</strong></div>
              <Button onClick={signOutGoogle} className="bg-gray-200 text-gray-900 hover:bg-gray-300">Sign out</Button>
            </div>
          )}
        </div>
      </div>

      {/* CHAT */}
      <div className="w-full max-w-3xl bg-white border border-gray-200 rounded-2xl p-4 flex flex-col shadow-lg h-[70vh]">
        <ScrollArea className="flex-1 pr-3">
          <div className="flex flex-col gap-3">
            {messages.map((msg, idx) => (
              <div key={idx} className={`my-2 flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`px-4 py-3 max-w-[75%] rounded-2xl text-sm leading-relaxed shadow-sm ${msg.sender === "user" ? "bg-blue-600 text-white rounded-br-none" : "bg-gray-50 text-gray-800 border border-gray-200 rounded-bl-none"}`}>
                  {msg.sender === "bot" ? (
                    <div className="prose max-w-none break-words">
                      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                        {msg.text}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <div className="whitespace-pre-wrap">{msg.text}</div>
                  )}
                </div>
              </div>
            ))}

            {isTyping && (
              <div className="flex justify-start">
                <div className="px-4 py-3 bg-gray-50 border border-gray-200 rounded-2xl text-gray-600">
                  <span className="animate-pulse">Typing...</span>
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>
        </ScrollArea>

        {/* input area */}
        <div className="mt-4 flex items-center gap-2">
          <Input placeholder="Type your message..." className="flex-1 bg-white border border-gray-300 text-gray-900" value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !isTyping) sendMessage(); }} />

          {isTyping ? (
            <Button onClick={stopResponse} className="px-6 bg-red-600 hover:bg-red-700 text-white">Stop</Button>
          ) : (
            <Button onClick={sendMessage} className="px-6 bg-green-600 hover:bg-green-700 text-white">Send</Button>
          )}
        </div>
      </div>
    </div>
  );
}