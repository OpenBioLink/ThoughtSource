## ThoughtSource⚡ Annotator Backend

Flask app to support the ThoughtSource⚡ Annotator react app. Start using
```
flask run
```

Provides sentence-to-sentence similarities to help the frontend visualise similar sentences between different CoT outputs.

Stores user annotations per session, to restore them in case of browser crashes and the like. Sessions and annotations are stored in the server's cache, furthermore they are written to disk on exit (and re-read at startup).

! Careful - Debug mode calls save to file twice, overwrites with empty file the second time around.
