import unittest
import pickle
import os
import hmac
import hashlib
import time
from unittest.mock import patch

# Assuming the sglang package is in the PYTHONPATH
from sglang.srt.distributed.device_communicators.shm_broadcast import (
    MessageQueue,
    Handle,
    ShmRingBuffer,
)

# For simulating ProcessGroup if direct instantiation is complex
# from torch.distributed import ProcessGroup # Not strictly needed if we manually create Handles

# Constants
DEFAULT_MAX_CHUNK_BYTES = 1024 * 10  # 10KB
DEFAULT_MAX_CHUNKS = 10
SIGNATURE_SIZE = 32  # SHA256

class TestShmBroadcastAuth(unittest.TestCase):
    def setUp(self):
        self.secret_key = os.urandom(32)
        self.writer_rank = 0
        self.local_reader_rank = 1
        self.remote_reader_rank = 2 # Ensure this is different from local_reader_rank

        # Clean up any stray shared memory from previous runs if necessary
        # This is a bit hacky; usually, __del__ should handle it, but good for robust testing
        for i in range(DEFAULT_MAX_CHUNKS * 2): # Check a few potential names
            try:
                shm = ShmRingBuffer(1, DEFAULT_MAX_CHUNK_BYTES, DEFAULT_MAX_CHUNKS, name=f"test_shm_{i}")
                shm.shared_memory.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass


    def tearDown(self):
        # Ensure all queues are properly closed and resources are released
        # hasattr check is important as some tests might fail before these are initialized
        if hasattr(self, "writer_mq") and self.writer_mq:
            del self.writer_mq
        if hasattr(self, "local_reader_mq") and self.local_reader_mq:
            del self.local_reader_mq
        if hasattr(self, "remote_reader_mq") and self.remote_reader_mq:
            del self.remote_reader_mq
        
        # Give a small delay to ensure zmq sockets can close, especially in CI
        time.sleep(0.1)


    def _setup_writer_local_reader(self, n_local_reader=1, max_chunk_bytes=DEFAULT_MAX_CHUNK_BYTES, max_chunks=DEFAULT_MAX_CHUNKS):
        local_reader_ranks = list(range(self.writer_rank + 1, self.writer_rank + 1 + n_local_reader))
        
        self.writer_mq = MessageQueue(
            n_reader=n_local_reader, 
            n_local_reader=n_local_reader,
            local_reader_ranks=local_reader_ranks,
            secret_key=self.secret_key,
            max_chunk_bytes=max_chunk_bytes,
            max_chunks=max_chunks
        )
        writer_handle = self.writer_mq.export_handle()

        self.local_reader_mqs = []
        for rank in local_reader_ranks:
            reader_mq = MessageQueue.create_from_handle(writer_handle, rank)
            self.local_reader_mqs.append(reader_mq)

        self.writer_mq.wait_until_ready()
        for reader_mq in self.local_reader_mqs:
            reader_mq.wait_until_ready()
        
        # Return the first local reader for convenience in single-reader tests
        return self.writer_mq, self.local_reader_mqs[0] if self.local_reader_mqs else None


    def _setup_writer_remote_reader(self, n_remote_reader=1, max_chunk_bytes=DEFAULT_MAX_CHUNK_BYTES, max_chunks=DEFAULT_MAX_CHUNKS):
        # Remote readers are not in local_reader_ranks.
        # Assign ranks that are clearly distinct from any local setup.
        remote_reader_start_rank = self.writer_rank + 10 
        
        self.writer_mq = MessageQueue(
            n_reader=n_remote_reader,
            n_local_reader=0, # No local readers for this specific setup
            local_reader_ranks=[],
            secret_key=self.secret_key,
            max_chunk_bytes=max_chunk_bytes,
            max_chunks=max_chunks
        )
        writer_handle = self.writer_mq.export_handle()

        self.remote_reader_mqs = []
        for i in range(n_remote_reader):
            rank = remote_reader_start_rank + i
            reader_mq = MessageQueue.create_from_handle(writer_handle, rank)
            self.remote_reader_mqs.append(reader_mq)
            
        self.writer_mq.wait_until_ready()
        for reader_mq in self.remote_reader_mqs:
            reader_mq.wait_until_ready()
            
        return self.writer_mq, self.remote_reader_mqs[0] if self.remote_reader_mqs else None

    # Test 1: Successful Local Communication (Shared Memory - No Overflow)
    def test_successful_local_no_overflow(self):
        writer, reader = self._setup_writer_local_reader()
        original_obj = {"message": "hello_local_no_overflow", "data": list(range(10))}
        
        writer.enqueue(original_obj)
        received_obj = reader.dequeue()
        
        self.assertEqual(received_obj, original_obj)

    # Test 2: Successful Local Communication (Shared Memory - With Overflow)
    def test_successful_local_with_overflow(self):
        # Force overflow: signature (32) + pickled_obj + overflow_byte (1) > max_chunk_bytes
        # Smallest pickled object is probably around 20-30 bytes for a simple dict.
        # So, if max_chunk_bytes is, e.g., 50, it should overflow.
        # len(pickle.dumps({"a":1})) is ~22 bytes.
        # 32 (sig) + 22 (obj) = 54.  Needs to be >= max_chunk_bytes - 1.
        # So if max_chunk_bytes = 50, 54 >= 49, which is true.
        writer, reader = self._setup_writer_local_reader(max_chunk_bytes=50) 
        original_obj = {"message": "hello_local_overflow", "data": list(range(20))} # Make it a bit larger
        
        writer.enqueue(original_obj)
        received_obj = reader.dequeue()
        
        self.assertEqual(received_obj, original_obj)

    # Test 3: Successful Remote Communication (ZeroMQ)
    def test_successful_remote_communication(self):
        writer, reader = self._setup_writer_remote_reader()
        original_obj = {"message": "hello_remote", "data": list(range(10))}
        
        writer.enqueue(original_obj)
        received_obj = reader.dequeue()
        
        self.assertEqual(received_obj, original_obj)

    # Test 4: Failed Local Communication - Tampered Data (Shared Memory - No Overflow)
    def test_failed_local_tampered_data_no_overflow(self):
        writer, reader = self._setup_writer_local_reader()
        original_obj = {"message": "tamper_test_no_overflow", "data": [1,2,3]}
        writer.enqueue(original_obj)

        # Tamper with the data in shared memory
        # This requires knowledge of the buffer layout:
        # buf[0] = overflow_flag (0 for no overflow)
        # buf[1:1+SIGNATURE_SIZE] = signature
        # buf[1+SIGNATURE_SIZE:] = serialized_obj
        with writer.buffer.get_data(writer.current_idx -1 % writer.buffer.max_chunks) as buf: # -1 because current_idx moved to next
            # Corrupt a byte in the serialized object part
            # Ensure we don't corrupt the signature or overflow byte
            if len(buf) > 1 + SIGNATURE_SIZE + 5: # Check if there's enough space to tamper
                 buf[1 + SIGNATURE_SIZE + 2] = buf[1 + SIGNATURE_SIZE + 2] ^ 0xFF # Flip some bits
            else:
                self.skipTest("Buffer too small to reliably tamper for this test.")
        
        with self.assertRaises(ValueError) as context:
            reader.dequeue()
        self.assertIn("HMAC signature verification failed", str(context.exception))

    # Test 5: Failed Local Communication - Tampered Data (Shared Memory - With Overflow)
    @patch.object(MessageQueue, 'local_socket') # We need to mock the socket on the *writer* instance
    def test_failed_local_tampered_data_overflow(self, mock_socket_attr):
        # This test is a bit tricky. We need to mock the writer's local_socket.send method
        # The mock_socket_attr will give us a MagicMock for local_socket if accessed on an instance
        
        # Setup writer and reader
        writer, reader = self._setup_writer_local_reader(max_chunk_bytes=50) # Force overflow
        
        # We need to ensure the mock is associated with the *writer's* socket
        # The helper _setup_writer_local_reader creates self.writer_mq
        # So, we need to patch self.writer_mq.local_socket.send
        
        original_obj = {"message": "tamper_test_overflow", "data": list(range(30))} # Large object
        
        original_send = self.writer_mq.local_socket.send # Keep original for other calls if any
        
        def tampered_send(data):
            signature = data[:SIGNATURE_SIZE]
            serialized_obj = data[SIGNATURE_SIZE:]
            
            # Tamper: modify the serialized object
            corrupted_serialized_obj = bytearray(serialized_obj)
            if len(corrupted_serialized_obj) > 5:
                 corrupted_serialized_obj[5] = corrupted_serialized_obj[5] ^ 0xFF
            else: # if object too small, just prepend garbage
                corrupted_serialized_obj = b"bad" + corrupted_serialized_obj

            tampered_data = signature + bytes(corrupted_serialized_obj)
            return original_send(tampered_data)

        # Patch the send method of the writer's local_socket
        with patch.object(self.writer_mq.local_socket, 'send', side_effect=tampered_send) as mock_send:
            writer.enqueue(original_obj) # This will call the tampered_send

            with self.assertRaises(ValueError) as context:
                reader.dequeue()
            self.assertIn("HMAC signature verification failed", str(context.exception))
            mock_send.assert_called_once() # Ensure our mock was actually used

    # Test 6: Failed Remote Communication - Tampered Signature (ZeroMQ)
    def test_failed_remote_tampered_signature(self):
        writer, reader = self._setup_writer_remote_reader()
        original_obj = {"message": "remote_tamper_sig", "data": "test"}
        writer.enqueue(original_obj)

        # Mock recv_multipart on the reader's remote_socket
        original_recv_multipart = reader.remote_socket.recv_multipart
        
        def tampered_recv_multipart():
            parts = original_recv_multipart()
            signature, serialized_obj = parts[0], parts[1]
            
            # Tamper with the signature
            corrupted_signature = bytearray(signature)
            corrupted_signature[0] = corrupted_signature[0] ^ 0xFF # Flip a bit
            return [bytes(corrupted_signature), serialized_obj]

        with patch.object(reader.remote_socket, 'recv_multipart', side_effect=tampered_recv_multipart):
            with self.assertRaises(ValueError) as context:
                reader.dequeue()
            self.assertIn("HMAC signature verification failed", str(context.exception))
            
    # Test 7: Failed Remote Communication - Tampered Data (ZeroMQ)
    def test_failed_remote_tampered_data(self):
        writer, reader = self._setup_writer_remote_reader()
        original_obj = {"message": "remote_tamper_data", "data": "test_data"}
        writer.enqueue(original_obj)

        original_recv_multipart = reader.remote_socket.recv_multipart
        
        def tampered_recv_multipart():
            parts = original_recv_multipart()
            signature, serialized_obj = parts[0], parts[1]
            
            # Tamper with the data
            corrupted_data = bytearray(serialized_obj)
            if len(corrupted_data) > 2:
                corrupted_data[2] = corrupted_data[2] ^ 0xFF # Flip a bit
            else:
                corrupted_data = b"tampered" + corrupted_data
            return [signature, bytes(corrupted_data)]

        with patch.object(reader.remote_socket, 'recv_multipart', side_effect=tampered_recv_multipart):
            with self.assertRaises(ValueError) as context:
                reader.dequeue()
            self.assertIn("HMAC signature verification failed", str(context.exception))

    # Test 8: Communication with no readers
    def test_communication_with_no_readers(self):
        # Setup writer with n_reader = 0
        self.writer_mq = MessageQueue(
            n_reader=0,
            n_local_reader=0,
            local_reader_ranks=[],
            secret_key=None # Key should not be generated if no readers
        )
        self.assertIsNone(self.writer_mq.secret_key) # Verify key is None
        
        original_obj = {"message": "no_readers_test"}
        try:
            # Enqueue should not fail and ideally not perform HMAC ops
            self.writer_mq.enqueue(original_obj) 
        except Exception as e:
            self.fail(f"enqueue failed with no readers: {e}")
        
        # No readers to dequeue, so the main check is that enqueue doesn't error
        # and that secret_key was not generated.
        # Also, check if sockets are None if no readers
        self.assertIsNone(self.writer_mq.local_socket)
        self.assertIsNone(self.writer_mq.remote_socket)


if __name__ == "__main__":
    unittest.main()
