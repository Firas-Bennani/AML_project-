import Sidebar from './Sidebar';

function AppLayout({ children }) {
  return (
    <div className="flex min-h-screen bg-gray-950">
      <Sidebar />
      {/* Main content pushed right by sidebar width */}
      <main className="flex-1 ml-56 min-h-screen overflow-y-auto">{children}</main>
    </div>
  );
}

export default AppLayout;
