import { useEffect, useRef } from 'react';

import { mountAvatarDemo } from './avatar';

import './avatar.css';

function App() {
    const root = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!root.current) return;
        return mountAvatarDemo(root.current);
    }, []);

    return <div className="avatar-demo-root" ref={root} />;
}

export default App;
