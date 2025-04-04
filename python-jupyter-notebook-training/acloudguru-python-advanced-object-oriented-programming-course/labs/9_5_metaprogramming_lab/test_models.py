import unittest

from models import ModelClass


class Post(metaclass=ModelClass):
    __tablename__ = "posts"

    title: str
    content: str
    published: bool


class OtherExample(metaclass=ModelClass):
    title: str
    content: str
    published: bool


class TestModelClass(unittest.TestCase):
    def test_generated_init(self):
        post = Post("New Title", "This is my content", False)

        self.assertEqual(post.__tablename__, "posts")
        self.assertEqual(post.title, "New Title")
        self.assertEqual(post.content, "This is my content")
        self.assertEqual(post.published, False)

    def test_default_tablename(self):
        self.assertEqual(OtherExample.__tablename__, "other_example")

    def test_generated_get_by_methods(self):
        info = dir(Post)
        self.assertTrue("get_by_title" in info)
        self.assertTrue("get_by_content" in info)
        self.assertTrue("get_by_published" in info)

        self.assertEqual(
            Post.get_by_title("My Title"),
            "select * from posts where title = 'My Title'",
        )
        self.assertEqual(
            Post.get_by_content("Some content"),
            "select * from posts where content = 'Some content'",
        )
        self.assertEqual(
            Post.get_by_published(False), "select * from posts where published = false"
        )

