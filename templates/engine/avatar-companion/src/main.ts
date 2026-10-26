import { mountAvatarDemo } from './avatar';

import './avatar.css';

const root = document.getElementById('root');

if (!root) {
    throw new Error('Avatar Companion requires a #root element.');
}

mountAvatarDemo(root);
