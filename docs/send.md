# Sending plots by email

To send things by email with dreamcoat you will first need to set up a Gmail account with an app-specific password.  This account is used to send the emails and its username goes into the `gmail_username` kwarg below.

To attach all files with a given path and extension to an email and send it, use

```python
import dreamcoat as dc

dc.send.all_files_from_dir(
    filepath=".",
    extension=".png",
    gmail_username=None,
    separate=False,
    contents=None,
    subject="[dreamcoat] all_files_from_dir",
    to=None,
)
```

  * Only files matching the specified `extension` are sent.
  * `separate` determines whether to send each file in a separate email or attach them all to the same one.
  * [Yagmail](https://github.com/kootenpv/yagmail) does the work behind the scenes, so see its documentation for more info about the other kwargs.