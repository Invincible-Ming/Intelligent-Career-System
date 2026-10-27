export default function (component) {
    const {data, parentElement} = component;
    const form = parentElement.querySelector("form");
    const status = parentElement.querySelector(".status");
    const retry = parentElement.querySelector(".retry");
    const button = form.querySelector("button[type='submit']");
    const passwordToggle = form.querySelector(".password-toggle");
    let controller;
    let active = true;
    let submitting = false;
    let reloadStarted = false;
    let requestTimer;
    const loggingOut = data.mode === "logout";
    form.hidden = loggingOut;

    async function submit() {
        if (submitting || reloadStarted) return;
        submitting = true;
        controller = new AbortController();
        button.disabled = true;
        form.setAttribute("aria-busy", "true");
        button.textContent = loggingOut ? "正在退出…" : "正在登录…";
        retry.hidden = true;
        // 登录进度已经显示在提交按钮上，状态行留空，避免同一句话出现两次；
        // 退出登录时表单整体隐藏，只剩这一行可以提示进度。
        status.hidden = !loggingOut;
        status.classList.remove("error");
        status.textContent = loggingOut ? "正在退出登录…" : "";
        const body = new URLSearchParams({return_to: data.return_to});
        if (!loggingOut) {
            body.set("username", form.elements.username.value);
            body.set("password", form.elements.password.value);
        }
        try {
            requestTimer = setTimeout(() => controller.abort(), 20000);
            // The response never contains the token. The browser stores Set-Cookie.
            const response = await fetch(data.url, {
                method: "POST", credentials: "include", cache: "no-store", signal: controller.signal,
                headers: {Accept: "application/json", "Content-Type": "application/x-www-form-urlencoded"},
                body
            });
            const result = await response.json();
            if (!active) return;
            if (!response.ok || !(loggingOut ? result.cleared : result.logged_in)) {
                throw new Error(typeof result.detail === "string" ? result.detail : "请检查输入后重试");
            }
            if (!loggingOut) form.elements.password.value = "";
            // A new WebSocket request allows Streamlit to receive the HttpOnly cookie.
            reloadStarted = true;
            window.location.reload();
        } catch (error) {
            if (!active) return;
            status.hidden = false;
            status.classList.add("error");
            status.textContent = error.name === "AbortError" ? "连接超时，请重试。"
                : error instanceof TypeError ? "无法连接后端，请检查连接后重试。" : error.message;
            retry.hidden = !loggingOut;
        } finally {
            clearTimeout(requestTimer);
            if (active && !reloadStarted) {
                submitting = false;
                form.removeAttribute("aria-busy");
                button.disabled = false;
                button.textContent = "登录";
            }
        }
    }

    const togglePassword = () => {
        const password = form.elements.password;
        const reveal = password.type === "password";
        password.type = reveal ? "text" : "password";
        passwordToggle.textContent = reveal ? "隐藏" : "显示";
        passwordToggle.setAttribute("aria-pressed", String(reveal));
        passwordToggle.setAttribute("aria-label", reveal ? "隐藏密码" : "显示密码");
    };

    const onSubmit = event => {
        event.preventDefault();
        submit();
    };
    form.addEventListener("submit", onSubmit);
    passwordToggle.addEventListener("click", togglePassword);
    retry.addEventListener("click", submit);
    if (loggingOut) submit();
    return () => {
        active = false;
        clearTimeout(requestTimer);
        controller?.abort();
        form.removeEventListener("submit", onSubmit);
        passwordToggle.removeEventListener("click", togglePassword);
        retry.removeEventListener("click", submit);
    };
}
